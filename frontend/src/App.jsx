
import { useMemo, useState } from "react";
import {
  Activity, AlertTriangle, Bell, Camera, CheckCircle2, ChevronRight,
  CircleGauge, Clock3, CloudUpload, Cpu, Database, FileVideo,
  Gauge, LayoutDashboard, Menu, Radio, Search, Server, Settings,
  ShieldCheck, Signal, Thermometer, Video, Waves, X, Zap
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip,
  XAxis, YAxis, BarChart, Bar
} from "recharts";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const navigation = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "video", label: "Video Analysis", icon: Video },
  { id: "live", label: "Live Monitoring", icon: Radio },
  { id: "sensors", label: "Sensor Analytics", icon: Activity },
  { id: "incidents", label: "Incident History", icon: AlertTriangle },
  { id: "model", label: "Model Information", icon: Cpu }
];

const sensorData = [
  { time: "10:00", vibration: 18, acoustic: 24, occupancy: 35 },
  { time: "10:05", vibration: 22, acoustic: 28, occupancy: 42 },
  { time: "10:10", vibration: 20, acoustic: 25, occupancy: 38 },
  { time: "10:15", vibration: 35, acoustic: 42, occupancy: 52 },
  { time: "10:20", vibration: 28, acoustic: 31, occupancy: 47 },
  { time: "10:25", vibration: 24, acoustic: 29, occupancy: 40 },
  { time: "10:30", vibration: 30, acoustic: 36, occupancy: 49 }
];

const sampleIncidents = [
  { id: "INC-001", type: "Sample incident", location: "Demo Camera 01", time: "10:15 AM", status: "Sample" },
  { id: "INC-002", type: "Sample incident", location: "Demo Camera 02", time: "09:42 AM", status: "Sample" },
  { id: "INC-003", type: "Sample incident", location: "Demo Camera 03", time: "Yesterday", status: "Sample" }
];

function MetricCard({ title, value, detail, icon: Icon, color }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <div className={`metric-icon ${color}`}><Icon size={19} /></div>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}

function SectionTitle({ title, subtitle, children }) {
  return (
    <div className="section-title">
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      {children}
    </div>
  );
}

function Overview({ setPage }) {
  return (
    <>
      <SectionTitle title="Overview" subtitle="Traffic intelligence and incident monitoring overview">
        <span className="date-label"><Clock3 size={15} /> System dashboard</span>
      </SectionTitle>

      <div className="notice">
        <ShieldCheck size={19} />
        <div>
          <strong>Prototype environment</strong>
          <span>Dashboard analytics below are illustrative demo values, not verified live detections.</span>
        </div>
      </div>

      <div className="metrics-grid">
        <MetricCard title="Video Streams" value="01" detail="Local test video available" icon={Camera} color="blue" />
        <MetricCard title="Incidents Detected" value="—" detail="No validated detections yet" icon={AlertTriangle} color="red" />
        <MetricCard title="Sensor Channels" value="06" detail="Synthetic CSV test input" icon={Activity} color="purple" />
        <MetricCard title="Model Status" value="Prototype" detail="Real-world training pending" icon={Cpu} color="green" />
      </div>

      <div className="content-grid">
        <div className="panel chart-panel">
          <div className="panel-heading">
            <div><h3>Sensor activity</h3><p>Illustrative sample readings</p></div>
            <span className="demo-tag">DEMO DATA</span>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sensorData}>
                <defs>
                  <linearGradient id="vibrationFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#36a3ff" stopOpacity={0.28} />
                    <stop offset="95%" stopColor="#36a3ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#263448" strokeDasharray="4 4" />
                <XAxis dataKey="time" stroke="#8290a5" tickLine={false} axisLine={false} />
                <YAxis stroke="#8290a5" tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "#111c2b", border: "1px solid #2a3a50", borderRadius: 10 }} />
                <Area type="monotone" dataKey="vibration" stroke="#36a3ff" fill="url(#vibrationFill)" strokeWidth={2} />
                <Area type="monotone" dataKey="acoustic" stroke="#a78bfa" fill="transparent" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-legend"><span><i className="blue-dot" /> Vibration</span><span><i className="purple-dot" /> Acoustic</span></div>
        </div>

        <div className="panel">
          <div className="panel-heading"><div><h3>System architecture</h3><p>Processing pipeline</p></div></div>
          <div className="pipeline">
            {["CCTV + IoT Inputs", "CNN + LSTM Extraction", "Attention Fusion", "Incident Classifier", "Adaptive Threshold"].map((item, index) => (
              <div className="pipeline-item" key={item}>
                <span className="pipeline-number">{index + 1}</span>
                <span>{item}</span>
                {index < 4 && <ChevronRight size={16} />}
              </div>
            ))}
          </div>
          <button className="secondary-button full-button" onClick={() => setPage("model")}>Explore architecture <ChevronRight size={16} /></button>
        </div>
      </div>

      <div className="panel bottom-panel">
        <div className="panel-heading"><div><h3>Quick actions</h3><p>Start working with your traffic data</p></div></div>
        <div className="quick-actions">
          <button onClick={() => setPage("video")}><CloudUpload size={19} /> Analyse a video</button>
          <button onClick={() => setPage("sensors")}><Activity size={19} /> View sensor analytics</button>
          <button onClick={() => setPage("incidents")}><AlertTriangle size={19} /> View incident history</button>
        </div>
      </div>
    </>
  );
}

function VideoAnalysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_URL}/analyze-video`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Video analysis failed.");
      setResult(data);
    } catch (err) {
      setError(`${err.message}. Check that the FastAPI backend is running and its URL is configured.`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <SectionTitle title="Video Analysis" subtitle="Upload CCTV footage for backend analysis" />
      <div className="panel upload-panel">
        <div className="upload-icon"><FileVideo size={32} /></div>
        <h2>Upload traffic video</h2>
        <p>Select a video file from your computer to send it to the Python backend.</p>
        <label className="file-picker">
          <CloudUpload size={18} /> Choose video
          <input type="file" accept="video/*" onChange={e => { setFile(e.target.files?.[0] || null); setResult(null); setError(""); }} />
        </label>
        {file && <div className="selected-file"><FileVideo size={17} /> {file.name} <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>}
        <button className="primary-button analyze-button" disabled={!file || loading} onClick={analyze}>
          {loading ? <><span className="spinner" /> Processing video...</> : <><Zap size={17} /> Start analysis</>}
        </button>
        <p className="small-note">Video processing happens on the backend, not in the browser.</p>
      </div>

      {error && <div className="error-box"><AlertTriangle size={18} />{error}</div>}
      {result && (
        <div className="panel result-panel">
          <h3>Backend response</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </>
  );
}

function LiveMonitoring() {
  return (
    <>
      <SectionTitle title="Live Monitoring" subtitle="Camera and roadside sensor monitoring" />
      <div className="notice warning-notice">
        <Radio size={19} />
        <div><strong>Live feeds are not connected</strong><span>This interface is ready for integration with live CCTV streams and physical IoT devices.</span></div>
      </div>
      <div className="camera-grid">
        {["Camera 01", "Camera 02", "Camera 03"].map((camera, index) => (
          <div className="camera-card" key={camera}>
            <div className="camera-placeholder"><Camera size={35} /><span>NO LIVE FEED</span></div>
            <div className="camera-info"><div><strong>{camera}</strong><span>Stream {index + 1}</span></div><span className="offline"><i /> Offline</span></div>
          </div>
        ))}
      </div>
    </>
  );
}

function Sensors() {
  const channels = [
    { name: "Vibration", value: "—", unit: "Awaiting live input", icon: Activity },
    { name: "Acoustic", value: "—", unit: "Awaiting live input", icon: Waves },
    { name: "Occupancy", value: "—", unit: "Awaiting live input", icon: Gauge },
    { name: "Ultrasonic", value: "—", unit: "Awaiting live input", icon: Signal },
    { name: "Illumination", value: "—", unit: "Awaiting live input", icon: Zap },
    { name: "Rainfall", value: "—", unit: "Awaiting live input", icon: Thermometer }
  ];
  return (
    <>
      <SectionTitle title="Sensor Analytics" subtitle="Six IoT input channels used by the proposed model" />
      <div className="notice"><Database size={19} /><div><strong>Sensor data status</strong><span>The current project test CSV contains synthetic readings. Live sensor integration is pending.</span></div></div>
      <div className="sensor-grid">
        {channels.map(({ name, value, unit, icon: Icon }) => (
          <div className="sensor-card" key={name}><div className="metric-icon purple"><Icon size={19} /></div><span>{name}</span><strong>{value}</strong><small>{unit}</small></div>
        ))}
      </div>
      <div className="panel chart-panel">
        <div className="panel-heading"><div><h3>Example sensor trends</h3><p>Illustrative values for interface preview only</p></div><span className="demo-tag">DEMO DATA</span></div>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sensorData}>
              <CartesianGrid stroke="#263448" strokeDasharray="4 4" />
              <XAxis dataKey="time" stroke="#8290a5" />
              <YAxis stroke="#8290a5" />
              <Tooltip contentStyle={{ background: "#111c2b", border: "1px solid #2a3a50", borderRadius: 10 }} />
              <Bar dataKey="vibration" fill="#36a3ff" radius={[4,4,0,0]} />
              <Bar dataKey="acoustic" fill="#a78bfa" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}

function Incidents() {
  return (
    <>
      <SectionTitle title="Incident History" subtitle="Review incident records returned by the detection system" />
      <div className="notice warning-notice"><AlertTriangle size={19} /><div><strong>No validated incident history connected</strong><span>The records below are illustrative placeholders, not actual detections.</span></div></div>
      <div className="panel table-panel">
        <div className="panel-heading"><div><h3>Incident records</h3><p>Sample interface records</p></div><span className="demo-tag">SAMPLE</span></div>
        <div className="table-scroll"><table><thead><tr><th>Incident ID</th><th>Type</th><th>Camera / Location</th><th>Time</th><th>Status</th></tr></thead>
          <tbody>{sampleIncidents.map(item => <tr key={item.id}><td>{item.id}</td><td>{item.type}</td><td>{item.location}</td><td>{item.time}</td><td><span className="status-pill">{item.status}</span></td></tr>)}</tbody>
        </table></div>
      </div>
    </>
  );
}

function ModelInfo() {
  const stages = [
    ["01", "Visual feature extraction", "CNN", "Extracts spatial features from CCTV frames."],
    ["02", "Sensor sequence processing", "LSTM", "Learns temporal patterns from six sensor channels."],
    ["03", "Multimodal feature fusion", "Attention", "Combines visual and sensor representations."],
    ["04", "Incident classification", "Classifier", "Estimates the incident probability."],
    ["05", "Decision mechanism", "Adaptive threshold", "Applies a decision boundary to the prediction."]
  ];
  return (
    <>
      <SectionTitle title="Model Information" subtitle="Architecture and technical details of the proposed system" />
      <div className="panel model-intro"><div className="model-logo"><Cpu size={30} /></div><div><h2>Multimodal Traffic Incident Detection</h2><p>CCTV + IoT sensor intelligence architecture</p></div><span className="status-pill">Prototype</span></div>
      <div className="panel">
        <div className="panel-heading"><div><h3>Model pipeline</h3><p>Processing stages</p></div></div>
        <div className="model-stages">{stages.map(([number, title, technology, description]) => (
          <div className="model-stage" key={number}><span className="stage-number">{number}</span><div><strong>{title}</strong><p>{description}</p></div><span className="technology-pill">{technology}</span></div>
        ))}</div>
      </div>
      <div className="metrics-grid model-metrics">
        <MetricCard title="Visual feature size" value="128" detail="CNN output dimensions" icon={Camera} color="blue" />
        <MetricCard title="Sensor feature size" value="96" detail="LSTM output dimensions" icon={Activity} color="purple" />
        <MetricCard title="Fused feature size" value="128" detail="Attention fusion output" icon={Zap} color="green" />
        <MetricCard title="Sensor channels" value="06" detail="Input measurement types" icon={Database} color="red" />
      </div>
      <div className="notice warning-notice"><AlertTriangle size={19} /><div><strong>Model validation status</strong><span>The current prototype has not been validated on labelled real-world incident data. Paper-reported research metrics are not prototype measurements.</span></div></div>
    </>
  );
}

export default function App() {
  const [page, setPage] = useState("overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const active = useMemo(() => navigation.find(item => item.id === page), [page]);

  const content = {
    overview: <Overview setPage={setPage} />,
    video: <VideoAnalysis />,
    live: <LiveMonitoring />,
    sensors: <Sensors />,
    incidents: <Incidents />,
    model: <ModelInfo />
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand"><div className="brand-mark"><Activity size={23} /></div><div><strong>Traffic<span>AI</span></strong><small>INCIDENT INTELLIGENCE</small></div><button className="close-sidebar" onClick={() => setSidebarOpen(false)}><X size={19}/></button></div>
        <div className="nav-label">WORKSPACE</div>
        <nav>{navigation.map(({ id, label, icon: Icon }) => (
          <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => { setPage(id); setSidebarOpen(false); }}>
            <Icon size={18} /><span>{label}</span>{page === id && <span className="active-indicator" />}
          </button>
        ))}</nav>
        <div className="sidebar-bottom"><div className="system-card"><span className="system-dot" /><div><strong>Prototype system</strong><small>Local development</small></div></div><div className="sidebar-version">TRAFFIC INCIDENT DETECTION <span>v1.0.0</span></div></div>
      </aside>

      {sidebarOpen && <div className="mobile-overlay" onClick={() => setSidebarOpen(false)} />}
      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left"><button className="menu-button" onClick={() => setSidebarOpen(true)}><Menu size={21}/></button><div className="breadcrumb">Workspace <span>/</span> {active?.label}</div></div>
          <div className="topbar-right"><span className="environment"><i /> DEVELOPMENT</span><button className="icon-button" title="Notifications"><Bell size={18}/></button><div className="avatar">YM</div></div>
        </header>
        <div className="page-content">{content[page]}</div>
        <footer className="footer"><span>TrafficAI · Multimodal Traffic Incident Detection</span><span>Built with React + FastAPI</span></footer>
      </main>
    </div>
  );
}
