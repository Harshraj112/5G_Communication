import React, { useState, useEffect, useRef, useMemo } from 'react'
import './index.css'

const MAX_PTS = 200
const ROLL_W = 100
const CLRS = {
  embb: '#10b981',
  urllc: '#ef4444',
  mmtc: '#f59e0b',
  reward: '#818cf8',
  muted: '#94a3b8',
}

// PlotlyChart Component
function PlotlyChart({ traces, layout, height = 230 }) {
  const ref = useRef(null)

  useEffect(() => {
    if (!ref.current || !traces?.length) return

    const BASE_LAYOUT = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Outfit, sans-serif', color: CLRS.muted, size: 11 },
      margin: { l: 46, r: 12, t: 6, b: 40 },
      xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
      yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
      legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
    }

    window.Plotly.react(
      ref.current,
      traces,
      { ...BASE_LAYOUT, ...layout },
      { displayModeBar: false, responsive: true }
    )
  }, [traces, layout])

  return <div ref={ref} style={{ width: '100%', height }} />
}

export default function App() {
  const [frame, setFrame] = useState(null)
  const [conn, setConn] = useState(false)
  const [clock, setClock] = useState('—')
  const [hist, setHist] = useState({
    t: [],
    aE: [],
    aU: [],
    aM: [],
    rw: [],
    sla: [],
    dE: [],
    dU: [],
    dM: [],
  })

  // Smart backend URL selection
  // Try local first, then fallback to production
  const [backendUrl, setBackendUrl] = useState(null)

  useEffect(() => {
    const determineBackendUrl = async () => {
      // Try localhost first
      try {
        const response = await fetch('http://localhost:5001/', { signal: AbortSignal.timeout(2000) })
        if (response.ok) {
          console.log('✅ Local backend detected at http://localhost:5001')
          setBackendUrl('http://localhost:5001')
          return
        }
      } catch (err) {
        console.log('⚠️ Local backend not available, using production URL')
      }
      
      // Fallback to production URL
      const prodUrl = import.meta.env.VITE_BACKEND_URL || 'https://fiveg-backend.onrender.com'
      console.log('Using production backend:', prodUrl)
      setBackendUrl(prodUrl)
    }

    determineBackendUrl()
  }, [])

  // Get backend URL from environment variable (backup)
  const BACKEND_URL = backendUrl || import.meta.env.VITE_BACKEND_URL || 'https://fiveg-backend.onrender.com'

  // SSE Connection
  useEffect(() => {
    if (!backendUrl) return // Wait for backend URL to be determined

    let es

    function connect() {
      const url = `${backendUrl}/stream`
      console.log('Connecting to:', url)

      es = new EventSource(url)

      es.onmessage = (ev) => {
        try {
          const d = JSON.parse(ev.data)
          setConn(true)
          setFrame(d)
          setClock(new Date().toLocaleTimeString())

          // Update history
          setHist((p) => ({
            t: [...p.t, d.t ?? 0].slice(-MAX_PTS),
            aE: [...p.aE, d.alloc?.eMBB ?? 0.5].slice(-MAX_PTS),
            aU: [...p.aU, d.alloc?.URLLC ?? 0.3].slice(-MAX_PTS),
            aM: [...p.aM, d.alloc?.mMTC ?? 0.2].slice(-MAX_PTS),
            rw: [...p.rw, d.reward ?? 0].slice(-MAX_PTS),
            sla: [...p.sla, d.sla_violations ?? 0].slice(-MAX_PTS),
            dE: [...p.dE, d.demand?.eMBB ?? 0].slice(-MAX_PTS),
            dU: [...p.dU, d.demand?.URLLC ?? 0].slice(-MAX_PTS),
            dM: [...p.dM, d.demand?.mMTC ?? 0].slice(-MAX_PTS),
          }))
        } catch (err) {
          console.error('Parse error:', err)
        }
      }

      es.onerror = () => {
        console.warn('SSE error, reconnecting...')
        setConn(false)
        es.close()
        setTimeout(connect, 3000)
      }
    }

    connect()
    return () => es?.close()
  }, [backendUrl])

  // Chart data
  const allocTr = useMemo(
    () => [
      {
        x: hist.t,
        y: hist.aE,
        name: 'eMBB',
        type: 'scatter',
        mode: 'none',
        stackgroup: 'one',
        fillcolor: 'rgba(16,185,129,0.52)',
      },
      {
        x: hist.t,
        y: hist.aU,
        name: 'URLLC',
        type: 'scatter',
        mode: 'none',
        stackgroup: 'one',
        fillcolor: 'rgba(239,68,68,0.52)',
      },
      {
        x: hist.t,
        y: hist.aM,
        name: 'mMTC',
        type: 'scatter',
        mode: 'none',
        stackgroup: 'one',
        fillcolor: 'rgba(245,158,11,0.45)',
      },
    ],
    [hist.t, hist.aE, hist.aU, hist.aM]
  )

  const demandTr = useMemo(
    () => [
      {
        x: hist.t,
        y: hist.dE,
        name: 'eMBB demand',
        mode: 'lines',
        line: { color: CLRS.embb, width: 2 },
      },
      {
        x: hist.t,
        y: hist.dU,
        name: 'URLLC demand',
        mode: 'lines',
        line: { color: CLRS.urllc, width: 2 },
      },
      {
        x: hist.t,
        y: hist.dM,
        name: 'mMTC demand',
        mode: 'lines',
        line: { color: CLRS.mmtc, width: 2 },
      },
    ],
    [hist.t, hist.dE, hist.dU, hist.dM]
  )

  const rwTr = useMemo(
    () => [
      {
        x: hist.t,
        y: hist.rw,
        name: 'Rewards',
        mode: 'lines',
        line: { color: CLRS.reward, width: 2 },
      },
    ],
    [hist.t, hist.rw]
  )

  return (
    <div className="app">
      {/* Header */}
      <header>
        <div className="header-left">
          <div className="logo">📡</div>
          <div>
            <h1>5G Network Intelligence</h1>
            <p className="sub">Render Backend + Vercel Frontend</p>
          </div>
        </div>
        <div className="header-status">
          <div
            className="live-dot"
            style={{
              background: conn ? '#10b981' : '#ef4444',
              boxShadow: `0 0 10px ${conn ? '#10b981' : '#ef4444'}`,
            }}
          />
          <span style={{ fontWeight: 600, color: conn ? '#10b981' : '#ef4444' }}>
            {conn ? 'LIVE' : 'CONNECTING'}
          </span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>
            {frame?.t ?? '—'}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <div className="content">
        {!frame ? (
          <div className="loading">
            <div className="spinner" />
            <h2>Connecting to Backend...</h2>
            <p>Backend URL: {BACKEND_URL}</p>
            <p style={{ fontSize: '0.85rem', color: CLRS.muted }}>
              Make sure backend is running and CORS is configured
            </p>
          </div>
        ) : (
          <>
            {/* KPI Cards */}
            <div className="kpi-grid">
              <div className="kpi-card">
                <div className="kpi-label">eMBB Allocation</div>
                <div className="kpi-value" style={{ color: CLRS.embb }}>
                  {(frame?.alloc?.eMBB * 100).toFixed(1)}%
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">URLLC Allocation</div>
                <div className="kpi-value" style={{ color: CLRS.urllc }}>
                  {(frame?.alloc?.URLLC * 100).toFixed(1)}%
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">mMTC Allocation</div>
                <div className="kpi-value" style={{ color: CLRS.mmtc }}>
                  {(frame?.alloc?.mMTC * 100).toFixed(1)}%
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Reward</div>
                <div className="kpi-value" style={{ color: CLRS.reward }}>
                  {frame?.reward?.toFixed(3)}
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="charts-grid">
              <div className="chart-card">
                <h3>📊 Bandwidth Allocation</h3>
                <PlotlyChart
                  traces={allocTr}
                  layout={{ yaxis: { title: 'Fraction', range: [0, 1] } }}
                  height={280}
                />
              </div>
              <div className="chart-card">
                <h3>📈 Network Demand</h3>
                <PlotlyChart
                  traces={demandTr}
                  layout={{ yaxis: { title: 'Demand (Mbps)' } }}
                  height={280}
                />
              </div>
              <div className="chart-card">
                <h3>💰 Reward Curve</h3>
                <PlotlyChart traces={rwTr} layout={{ yaxis: { title: 'Reward' } }} height={280} />
              </div>
            </div>

            {/* Raw Data */}
            <div className="data-card">
              <h3>Raw Frame Data</h3>
              <pre>{JSON.stringify(frame, null, 2)}</pre>
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <footer>
        <span>5G Intelligence System · Render + Vercel</span>
        <span>{clock}</span>
      </footer>
    </div>
  )
}
