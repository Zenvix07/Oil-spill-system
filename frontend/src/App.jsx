import React, { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  Popup,
  Polygon,
} from "react-leaflet";

const API = "http://localhost:8000";

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function ScoreBar({ value }) {
  return (
    <div className="score-wrap">
      <div className="score-track">
        <div className="score-fill" style={{ width: `${value}%` }} />
      </div>
      <span>{value}%</span>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("dashboard");

  const vessels = result?.vessels || [];
  const origin = result?.origin || { lat: 10.5, lon: 72.5 };
  const track = result?.drift_track || [];

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch(`${API}/api/analyze`, {
        method: "POST",
        body: form,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const mapCenter = [origin.lat, origin.lon];

  const pseudoPolygon = useMemo(() => {
    if (!result) return [];
    return [
      [origin.lat + 0.08, origin.lon - 0.08],
      [origin.lat + 0.04, origin.lon + 0.14],
      [origin.lat - 0.04, origin.lon + 0.10],
      [origin.lat - 0.07, origin.lon - 0.04],
      [origin.lat, origin.lon - 0.14],
    ];
  }, [result, origin.lat, origin.lon]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">◈</div>
          <div>
            <div className="brand-name">OILTRACE</div>
            <div className="brand-sub">MARITIME INTELLIGENCE</div>
          </div>
        </div>

        <div className="nav-title">OPERATIONS</div>
        {[
          ["dashboard", "◉", "Command Center"],
          ["analysis", "⌁", "Satellite Analysis"],
          ["ais", "◇", "AIS Attribution"],
        ].map(([id, icon, label]) => (
          <button
            key={id}
            className={`nav-btn ${tab === id ? "active" : ""}`}
            onClick={() => setTab(id)}
          >
            <span>{icon}</span>{label}
          </button>
        ))}

        <div className="sidebar-bottom">
          <div className="system-card">
            <div className="live-dot" />
            <div>
              <b>AI ENGINE ONLINE</b>
              <span>POSEatSea / SAR</span>
            </div>
          </div>
          <div className="version">NTRO • SIH 2026 prototype</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <div className="eyebrow">NATIONAL MARITIME MONITORING</div>
            <h1>{tab === "dashboard" ? "Oil Spill Command Center" :
                tab === "analysis" ? "Satellite Analysis" : "AIS Attribution"}</h1>
          </div>

          <div className="top-actions">
            <label className="upload-btn">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              + Upload SAR
            </label>
            <button className="analyze-btn" disabled={!file || loading} onClick={analyze}>
              {loading ? "Running AI…" : "Analyze Image"}
            </button>
          </div>
        </header>

        {error && <div className="error">{error}</div>}

        {!result && (
          <section className="hero-card">
            <div>
              <div className="hero-kicker">SATELLITE → AI → AIS</div>
              <h2>Detect. Reconstruct. Attribute.</h2>
              <p>
                Upload a Sentinel-1/SAR image and OILTRACE will run the
                pretrained segmentation model, generate an oil-spill mask,
                estimate spill geometry, reconstruct a probable origin and
                rank nearby vessels.
              </p>
              <label className="dropzone">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
                <div className="drop-icon">⇧</div>
                <strong>{file ? file.name : "Drop a satellite image here"}</strong>
                <span>PNG, JPG, TIFF preview • model inference uses 512×512</span>
              </label>
            </div>
            <div className="pipeline">
              {["SAR IMAGE", "SEGMENTATION", "DRIFT HINDCAST", "AIS CORRELATION"].map((x, i) => (
                <div className="pipeline-item" key={x}>
                  <span>0{i + 1}</span>{x}
                </div>
              ))}
            </div>
          </section>
        )}

        {result && (
          <>
            <section className="stats-grid">
              <Stat label="SPILL STATUS" value={result.oil_detected ? "DETECTED" : "CLEAR"} sub="AI segmentation" />
              <Stat label="OIL CONFIDENCE" value={`${result.oil_confidence}%`} sub="mean oil-class probability" />
              <Stat label="EST. AREA" value={`${result.estimated_area_km2} km²`} sub="image-space estimate" />
              <Stat label="TOP SUSPECT" value={vessels[0]?.score ? `${vessels[0].score}%` : "—"} sub={vessels[0]?.name || "No candidate"} />
            </section>

            <section className="content-grid">
              <div className="map-card">
                <div className="card-head">
                  <div>
                    <span className="card-kicker">GEOSPATIAL VIEW</span>
                    <h3>Spill & Vessel Correlation</h3>
                  </div>
                  <span className="live-pill">● LIVE ANALYSIS</span>
                </div>

                <MapContainer center={mapCenter} zoom={7} className="map">
                  <TileLayer
                    attribution='&copy; OpenStreetMap contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  {pseudoPolygon.length > 0 && (
                    <Polygon positions={pseudoPolygon} pathOptions={{ color: "#ff513b", fillOpacity: 0.32 }} />
                  )}
                  <CircleMarker center={mapCenter} radius={9} pathOptions={{ color: "#ffb347", fillColor: "#ff513b", fillOpacity: 1 }}>
                    <Popup>Probable spill origin</Popup>
                  </CircleMarker>
                  {track.length > 1 && <Polyline positions={track} pathOptions={{ color: "#55c8ff", weight: 3, dashArray: "7 8" }} />}
                  {vessels.map((v) => (
                    <CircleMarker key={v.mmsi} center={[v.lat, v.lon]} radius={7}
                      pathOptions={{ color: v.score > 80 ? "#ff513b" : "#ffd166", fillColor: v.score > 80 ? "#ff513b" : "#ffd166", fillOpacity: 0.95 }}>
                      <Popup>
                        <b>{v.name}</b><br />
                        MMSI {v.mmsi}<br />
                        Suspect score: {v.score}%
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>

                <div className="map-legend">
                  <span><i className="legend oil" /> Detected slick</span>
                  <span><i className="legend origin" /> Probable origin</span>
                  <span><i className="legend vessel" /> AIS vessel</span>
                  <span><i className="legend drift" /> Drift path</span>
                </div>
              </div>

              <div className="side-stack">
                <div className="panel">
                  <div className="card-kicker">MODEL OUTPUT</div>
                  <h3>Segmentation</h3>
                  <div className="image-pair">
                    <div>
                      <img src={`${API}${result.overlay_url}`} />
                      <span>Overlay</span>
                    </div>
                    <div>
                      <img src={`${API}${result.mask_url}`} />
                      <span>Oil mask</span>
                    </div>
                  </div>
                  <div className="distribution">
                    {Object.entries(result.class_distribution).map(([k, v]) => (
                      <div key={k} className="dist-row">
                        <span>{k.replace("_", " ")}</span><b>{v}%</b>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="panel">
                  <div className="card-kicker">ORIGIN ESTIMATE</div>
                  <h3>{origin.lat.toFixed(4)}°, {origin.lon.toFixed(4)}°</h3>
                  <p className="muted">Backward drift track is shown as a demonstration layer. Connect Copernicus/NOAA currents and wind forcing for scientific hindcasting.</p>
                </div>
              </div>
            </section>

            <section className="panel vessel-panel">
              <div className="card-head">
                <div>
                  <div className="card-kicker">HISTORICAL AIS CORRELATION</div>
                  <h3>Potential source vessels</h3>
                </div>
                <span className="muted">Ranked by spatio-temporal evidence</span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>RANK</th><th>VESSEL</th><th>TYPE</th><th>DISTANCE</th>
                      <th>PROXIMITY</th><th>TEMPORAL</th><th>TRAJECTORY</th><th>SCORE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vessels.map((v, i) => (
                      <tr key={v.mmsi}>
                        <td className="rank">{i + 1}</td>
                        <td><b>{v.name}</b><small>{v.mmsi}</small></td>
                        <td>{v.type}</td>
                        <td>{v.distance_km} km</td>
                        <td>{v.proximity}%</td>
                        <td>{v.temporal}%</td>
                        <td>{v.trajectory}%</td>
                        <td><ScoreBar value={v.score} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        <footer>
          <span>OILTRACE • Research prototype</span>
          <span>Detection ≠ attribution. Review required before operational action.</span>
        </footer>
      </main>
    </div>
  );
}
