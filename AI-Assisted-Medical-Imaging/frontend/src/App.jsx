
import { useEffect, useRef, useState } from "react";

// Docker backend runs on port 8000
const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8765";

async function api(path, options) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      // Ignore JSON parsing errors
    }
    throw new Error(detail);
  }
  return res.json();
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function App() {
  const [cases, setCases] = useState([]);
  const [current, setCurrent] = useState(null);
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [opacity, setOpacity] = useState(0.8);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef(null);

  const refresh = () => api("/cases").then(setCases).catch(() => {});

  useEffect(() => {
    api("/health")
      .then(setHealth)
      .catch(() => setError(`Cannot reach the API at ${API}. Start the backend first.`));

    refresh();

    const timer = window.setInterval(() => {
      refresh();
    }, 5000);

    return () => window.clearInterval(timer);
  }, []);

  async function upload(file) {
    if (!file) return;

    setBusy(true);
    setError("");

    try {
      const body = new FormData();
      body.append("file", file);

      const c = await api("/predict", {
        method: "POST",
        body,
      });

      setCurrent(c);
      refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="top">
        <h1>Scan review workstation</h1>

        <div className="status">
          {health && (
            <span className={health.model_loaded ? "pill ok" : "pill warn"}>
              {health.model_loaded ? "Trained model loaded" : "Untrained demo model"}
            </span>
          )}

          {health?.orthanc && (
            <span className="pill ok">Orthanc connected</span>
          )}
        </div>
      </header>

      <div className="notice" role="note" style={{ margin: "16px 0", lineHeight: 1.6 }}>
        <strong>Educational research prototype only.</strong> This AI system is
        not a medical device and must not be used for clinical diagnosis,
        treatment decisions, or emergency care. Predictions and Grad-CAM
        visualizations may be inaccurate. All findings require independent
        assessment by qualified healthcare professionals. The model was trained
        using a limited dataset and has not been clinically validated.
      </div>

      <div className="layout">
        <aside className="side">
          <div
            className={`drop ${drag ? "over" : ""}`}
            onClick={() => fileRef.current.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              upload(e.dataTransfer.files[0]);
            }}
          >
            <strong>{busy ? "Analysing scan..." : "Add a scan"}</strong>

            <span>
              Drop a chest X-ray here, or click to browse.
              PNG, JPG and DICOM (.dcm) work.
            </span>

            <input
              ref={fileRef}
              type="file"
              accept=".png,.jpg,.jpeg,.dcm,.dicom"
              hidden
              onChange={(e) => {
                upload(e.target.files[0]);
                e.target.value = "";
              }}
            />
          </div>

          <h2>Case queue</h2>

          {cases.length === 0 && (
            <p className="muted">No cases yet. Add a scan to start.</p>
          )}

          <ul className="queue">
            {cases.map((c) => (
              <li key={c.id}>
                <button
                  className={current?.id === c.id ? "row active" : "row"}
                  onClick={() => setCurrent(c)}
                >
                  <span className="fname">{c.filename}</span>

                  <span className="meta">
                    {c.label} · {pct(c.confidence)}
                  </span>

                  <span className={c.status === "reviewed" ? "chip done" : "chip wait"}>
                    {c.status === "reviewed" ? "Reviewed" : "Needs review"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main className="main">
          {error && <div className="error" role="alert">{error}</div>}

          {!current ? (
            <div className="empty">
              <h2>No scan open</h2>
              <p>
                Add a scan to see the model's finding and the regions it focused on,
                then record your review.
              </p>
            </div>
          ) : (
            <CaseView
              key={current.id}
              c={current}
              opacity={opacity}
              setOpacity={setOpacity}
              onSaved={(u) => {
                setCurrent(u);
                refresh();
              }}
            />
          )}
        </main>
      </div>

      <footer>Research prototype. Not for clinical use.</footer>
    </div>
  );
}

function CaseView({ c, opacity, setOpacity, onSaved }) {
  const [decision, setDecision] = useState(c.decision || "agree");
  const [finalLabel, setFinalLabel] = useState(c.final_label || c.label);
  const [notes, setNotes] = useState(c.notes || "");
  const [reviewer, setReviewer] = useState(c.reviewer || "");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");

  // Ground-truth localization state
  const [localizationUrl, setLocalizationUrl] = useState("");
  const [localizationLoading, setLocalizationLoading] = useState(false);
  const [localizationError, setLocalizationError] = useState("");

  const [audit, setAudit] = useState([]);
  const [auditError, setAuditError] = useState("");

  useEffect(() => {
    let active = true;
    api(`/cases/${c.id}/audit`)
      .then((rows) => {
        if (active) setAudit(rows);
      })
      .catch((e) => {
        if (active) setAuditError(e.message);
      });
    return () => { active = false; };
  }, [c.id]);

  const reviewed = c.status === "reviewed";

  const isDicom =
    Boolean(c.is_dicom) ||
    /\.(dcm|dicom)$/i.test(c.filename || "");

  // Release previous image URL when component changes or unmounts
  useEffect(() => {
    return () => {
      if (localizationUrl) {
        URL.revokeObjectURL(localizationUrl);
      }
    };
  }, [localizationUrl]);

  async function loadLocalization() {
    setLocalizationLoading(true);
    setLocalizationError("");

    try {
      const response = await fetch(
        `${API}/cases/${c.id}/localization`
      );

      if (!response.ok) {
        let message = `Localization request failed (${response.status})`;

        try {
          const data = await response.json();
          message = data.detail || message;
        } catch {
          // Keep default message
        }

        throw new Error(message);
      }

      const blob = await response.blob();

      if (!blob.type.startsWith("image/")) {
        throw new Error("The server did not return a valid image.");
      }

      const imageUrl = URL.createObjectURL(blob);
      setLocalizationUrl(imageUrl);
    } catch (e) {
      setLocalizationError(e.message);
    } finally {
      setLocalizationLoading(false);
    }
  }

  async function save(e) {
    e.preventDefault();
    setSaving(true);
    setErr("");

    try {
      const label = decision === "agree" ? c.label : finalLabel;

      const u = await api(`/cases/${c.id}/review`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          decision,
          final_label: label,
          notes,
          reviewer: reviewer || "Doctor",
        }),
      });

      onSaved(u);
      try {
        const history = await api(`/cases/${c.id}/audit`);
        setAudit(history);
        setAuditError("");
      } catch (auditEx) {
        setAuditError(auditEx.message);
      }
    } catch (ex) {
      setErr(ex.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <div className="case-head">
        <h2>{c.filename}</h2>

        <span className={reviewed ? "chip done" : "chip wait"}>
          {reviewed ? "Reviewed" : "Needs review"}
        </span>
      </div>

      {!c.model_trained && (
        <div className="notice">
          This result comes from an untrained model, so the label and heatmap
          are not meaningful. Train the model and restart the backend.
        </div>
      )}

      <div className="viewers">
        <figure>
          <div className="stage">
            <img
              src={`${API}/cases/${c.id}/image`}
              alt="Original scan"
            />
          </div>
          <figcaption>Original scan</figcaption>
        </figure>

        <figure>
          <div className="stage">
            <img
              src={`${API}/cases/${c.id}/image`}
              alt=""
            />

            <img
              className="overlay"
              style={{ opacity }}
              src={`${API}/cases/${c.id}/heatmap`}
              alt="Grad-CAM heatmap"
            />
          </div>

          <figcaption>
            <label className="slider">
              Heatmap strength

              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
            </label>
          </figcaption>
        </figure>
      </div>

      {/* RSNA Ground Truth Localization */}
      {isDicom && (
        <section className="card">
          <h3>RSNA Ground Truth Localization</h3>

          <p className="muted">
            View radiologist-annotated pneumonia bounding boxes
            from the RSNA dataset.
          </p>

          <button
            type="button"
            className="primary"
            disabled={localizationLoading}
            onClick={loadLocalization}
          >
            {localizationLoading
              ? "Loading localization..."
              : localizationUrl
                ? "Refresh Ground Truth Localization"
                : "View Ground Truth Localization"}
          </button>

          {localizationError && (
            <div className="error" role="alert" style={{ marginTop: "12px" }}>
              {localizationError}
            </div>
          )}

          {localizationUrl && (
            <figure style={{ margin: "20px 0 0" }}>
              <img
                src={localizationUrl}
                alt="RSNA ground truth pneumonia bounding boxes"
                style={{
                  display: "block",
                  width: "100%",
                  maxWidth: "650px",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #ccc",
                  margin: "0 auto",
                }}
              />

              <figcaption style={{ textAlign: "center", marginTop: "10px" }}>
                RSNA radiologist-annotated ground-truth bounding boxes
              </figcaption>
            </figure>
          )}

          <p className="muted small" style={{ marginTop: "12px" }}>
            Note: These red boxes are dataset ground-truth annotations,
            not predictions generated by the AI model.
          </p>
        </section>
      )}

      <section className="card">
        <h3>Model finding</h3>

        <p className="finding">
          <strong>{c.label}</strong>{" "}
          <span className="muted">
            at {pct(c.confidence)} confidence
          </span>
        </p>

        <div className="bars">
          <Bar name="No Lung Opacity" v={c.prob_normal} />
          <Bar name="Pneumonia" v={c.prob_pneumonia} />
        </div>
      </section>

      <form className="card" onSubmit={save}>
        <h3>Doctor review</h3>

        <div className="radios">
          <label>
            <input
              type="radio"
              checked={decision === "agree"}
              onChange={() => setDecision("agree")}
            />
            Agree with the model ({c.label})
          </label>

          <label>
            <input
              type="radio"
              checked={decision === "override"}
              onChange={() => setDecision("override")}
            />
            Override the model
          </label>
        </div>

        {decision === "override" && (
          <label className="field">
            Final diagnosis

            <select
              value={finalLabel}
              onChange={(e) => setFinalLabel(e.target.value)}
            >
              <option value="NO_LUNG_OPACITY">NO_LUNG_OPACITY</option>
              <option value="PNEUMONIA">PNEUMONIA</option>
            </select>
          </label>
        )}

        <label className="field">
          Reviewer name

          <input
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
            placeholder="Dr. Name"
          />
        </label>

        <label className="field">
          Notes

          <textarea
            rows="3"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Findings, follow-up, reason for override"
          />
        </label>

        {err && <div className="error">{err}</div>}

        <button className="primary" disabled={saving}>
          {saving ? "Saving..." : reviewed ? "Update review" : "Save review"}
        </button>

        {reviewed && (
          <p className="muted small">
            Last reviewed by {c.reviewer} on{" "}
            {new Date(c.reviewed_at).toLocaleString()}.
          </p>
        )}
      </form>

      <section className="card">
        <h3>Audit history</h3>
        <p className="muted small">
          Recorded scan and reviewer actions for this case.
        </p>

        {auditError && <div className="error">{auditError}</div>}

        {audit.length === 0 ? (
          <p className="muted">No audit events available for this case.</p>
        ) : (
          <ol style={{ paddingLeft: "22px" }}>
            {audit.map((event) => (
              <li key={event.id} style={{ marginBottom: "12px" }}>
                <strong>{(event.action || "EVENT").replaceAll("_", " ")}</strong>
                <div className="muted small">
                  {event.timestamp
                    ? new Date(event.timestamp).toLocaleString()
                    : "Time unavailable"}
                </div>
                {event.reviewer && <div>Reviewer: {event.reviewer}</div>}
                {event.decision && <div>Decision: {event.decision}</div>}
                {event.final_label && <div>Final label: {event.final_label}</div>}
                {event.notes && <div>Notes: {event.notes}</div>}
              </li>
            ))}
          </ol>
        )}
      </section>
    </>
  );
}

function Bar({ name, v }) {
  return (
    <div className="bar">
      <span>{name}</span>

      <div className="track">
        <div className="fill" style={{ width: pct(v) }} />
      </div>

      <span className="num">{pct(v)}</span>
    </div>
  );
}
