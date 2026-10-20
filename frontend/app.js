// Helios Solar Frontend Engine
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSiteSelector();
  initSimulator();
  initModalLightbox();
});

// Site metadata database
const SITES_DATA = {
  14: {
    name: 'Site 14 · Bundoora Main Campus',
    campus: 'Bundoora (Melbourne)',
    lat: -37.718,
    lon: 145.051,
    kwp: 66.0,
    inverter_kw: 55.2,
    panels: '200x Trina 330W',
    inverters: '2x SolarEdge SE27.6K',
    optimal_tilt: 20,
    annual_yield: 784.0,
    flat_yield: 755.3,
    gain_pct: 3.81,
    pr_pct: 56.8,
    r2_weather: 0.948,
    r2_power: 0.907,
    rmse_power: 4.69,
    peak_shaved: 8.5
  },
  27: {
    name: 'Site 27 · Bundoora Solar Farm',
    campus: 'Bundoora (Melbourne)',
    lat: -37.718,
    lon: 145.051,
    kwp: 539.8,
    inverter_kw: 450.0,
    panels: '1240x Sunpower 435W',
    inverters: '5x ABB Central Inverters',
    optimal_tilt: 22,
    annual_yield: 792.4,
    flat_yield: 755.3,
    gain_pct: 4.91,
    pr_pct: 61.2,
    r2_weather: 0.948,
    r2_power: 0.912,
    rmse_power: 28.4,
    peak_shaved: 68.0
  },
  1: {
    name: 'Site 1 · Wodonga Campus',
    campus: 'Albury-Wodonga',
    lat: -36.111,
    lon: 146.848,
    kwp: 30.0,
    inverter_kw: 25.0,
    panels: 'Commercial Monocrystalline',
    inverters: 'SMA Sunny Tripower',
    optimal_tilt: 25,
    annual_yield: 820.5,
    flat_yield: 780.1,
    gain_pct: 5.18,
    pr_pct: 64.0,
    r2_weather: 0.942,
    r2_power: 0.895,
    rmse_power: 2.35,
    peak_shaved: 4.2
  },
  6: {
    name: 'Site 6 · Bendigo Campus',
    campus: 'Bendigo',
    lat: -36.779,
    lon: 144.301,
    kwp: 45.0,
    inverter_kw: 38.0,
    panels: 'Trina Solar Modules',
    inverters: 'SolarEdge Distributed',
    optimal_tilt: 24,
    annual_yield: 812.0,
    flat_yield: 774.2,
    gain_pct: 4.88,
    pr_pct: 62.5,
    r2_weather: 0.945,
    r2_power: 0.901,
    rmse_power: 3.42,
    peak_shaved: 6.1
  },
  41: {
    name: 'Site 41 · Mildura Campus',
    campus: 'Mildura (Mallee Sun Belt)',
    lat: -34.205,
    lon: 142.166,
    kwp: 50.0,
    inverter_kw: 42.0,
    panels: 'Tier-1 Mono PERC',
    inverters: 'Fronius Eco',
    optimal_tilt: 28,
    annual_yield: 915.2,
    flat_yield: 860.0,
    gain_pct: 6.42,
    pr_pct: 68.2,
    r2_weather: 0.952,
    r2_power: 0.920,
    rmse_power: 3.10,
    peak_shaved: 7.8
  },
  42: {
    name: 'Site 42 · Shepparton Campus',
    campus: 'Shepparton (Goulburn Valley)',
    lat: -36.380,
    lon: 145.406,
    kwp: 35.0,
    inverter_kw: 30.0,
    panels: 'High-Efficiency Crystalline',
    inverters: 'SMA Solar',
    optimal_tilt: 24,
    annual_yield: 830.4,
    flat_yield: 791.0,
    gain_pct: 4.98,
    pr_pct: 63.8,
    r2_weather: 0.946,
    r2_power: 0.904,
    rmse_power: 2.65,
    peak_shaved: 5.0
  }
};

let currentSiteKey = 14;

// 1. Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = `tab-${tab.getAttribute('data-tab')}`;

      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });
}

// 2. Site Selector Handler
function initSiteSelector() {
  const siteSelect = document.getElementById('site-select');
  if (!siteSelect) return;

  siteSelect.addEventListener('change', (e) => {
    currentSiteKey = parseInt(e.target.value, 10);
    updateSiteView(currentSiteKey);
  });
}

function updateSiteView(siteKey) {
  const site = SITES_DATA[siteKey];
  if (!site) return;

  // Update simulator capacity
  const capacityTag = document.querySelector('.sim-capacity-tag');
  if (capacityTag) {
    capacityTag.textContent = `Rated Capacity: ${site.kwp.toFixed(1)} kWp · Inverter AC Ceiling: ${site.inverter_kw.toFixed(1)} kW`;
  }

  // Trigger recalculation in simulator
  runSimulation();
}

// 3. Interactive Physics PV Simulator
function initSimulator() {
  const sliders = ['sim-ghi', 'sim-temp', 'sim-wind', 'sim-cloud', 'sim-elevation'];
  
  sliders.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', runSimulation);
    }
  });

  runSimulation();
}

function runSimulation() {
  const ghi = parseFloat(document.getElementById('sim-ghi')?.value || 750);
  const temp = parseFloat(document.getElementById('sim-temp')?.value || 28);
  const wind = parseFloat(document.getElementById('sim-wind')?.value || 3.5);
  const cloud = parseFloat(document.getElementById('sim-cloud')?.value || 15);
  const elevation = parseFloat(document.getElementById('sim-elevation')?.value || 55);

  // Update slider label text
  updateLabel('val-ghi', `${ghi} W/m²`);
  updateLabel('val-temp', `${temp.toFixed(1)} °C`);
  updateLabel('val-wind', `${wind.toFixed(1)} m/s`);
  updateLabel('val-cloud', `${cloud} %`);
  updateLabel('val-elevation', `${elevation} °`);

  const site = SITES_DATA[currentSiteKey] || SITES_DATA[14];
  const kwp = site.kwp;
  const inverterMax = site.inverter_kw;

  // Night condition
  if (elevation <= 0) {
    updatePowerOutput(0.0, kwp, inverterMax, 'Nighttime (Sun Below Horizon)', 'Inverters in standby mode', temp);
    return;
  }

  // 1. Faiman PV Cell Temperature Model
  // T_cell = T_air + GHI / (25 + 6.84 * wind)
  const u0 = 25.0;
  const u1 = 6.84;
  const tCell = temp + (ghi / (u0 + u1 * wind));
  const tempDelta = tCell - 25.0;
  
  // Power Temperature Derating: -0.38% per °C above 25°C
  const gamma = -0.0038;
  const thermalDerate = Math.max(0.6, 1.0 + gamma * tempDelta);
  const thermalLossPct = (1.0 - thermalDerate) * 100.0;

  // 2. Clear-sky theoretical GHI approximation
  // Clearsky GHI ~ 1050 * sin(elevation)
  const sinElev = Math.sin((elevation * Math.PI) / 180.0);
  const clearskyGhi = Math.max(50.0, 1050.0 * Math.pow(sinElev, 1.15));
  const kt = Math.min(1.3, ghi / clearskyGhi);

  // Theoretical clear sky DC power
  const csPower = kwp * (clearskyGhi / 1000.0);

  // 3. DC Potential Generation
  let dcPotential = kwp * (ghi / 1000.0) * thermalDerate;

  // Inverter AC Clipping limit
  let acPower = dcPotential;
  let statusText = 'Active Generating';
  let statusMeta = 'Normal DC-to-AC conversion';

  if (acPower >= inverterMax) {
    acPower = inverterMax;
    statusText = 'Inverter Clipping Saturation';
    statusMeta = `Generation limited to ${inverterMax.toFixed(1)} kW AC limit`;
  }

  // Update UI Elements
  updatePowerOutput(acPower, kwp, inverterMax, statusText, statusMeta, tCell, thermalLossPct, csPower, kt);
}

function updatePowerOutput(powerKwh, kwp, inverterMax, status, statusMeta, tCell, thermalLossPct = 0, csPower = 0, kt = 0) {
  const powerEl = document.getElementById('res-power');
  const barEl = document.getElementById('res-power-bar');
  const tcellEl = document.getElementById('res-tcell');
  const tempLossEl = document.getElementById('res-temp-loss');
  const csPowerEl = document.getElementById('res-cs-power');
  const ktEl = document.getElementById('res-kt');
  const cloudImpactEl = document.getElementById('res-cloud-impact');
  const statusEl = document.getElementById('res-status');
  const statusMetaEl = document.getElementById('res-status-meta');

  if (powerEl) powerEl.textContent = powerKwh.toFixed(1);
  if (barEl) {
    const pct = Math.min(100, Math.max(0, (powerKwh / inverterMax) * 100));
    barEl.style.width = `${pct}%`;
  }

  if (tcellEl) tcellEl.textContent = `${tCell.toFixed(1)} °C`;
  if (tempLossEl) {
    tempLossEl.textContent = thermalLossPct >= 0 
      ? `-${thermalLossPct.toFixed(2)}% thermal loss` 
      : `+${Math.abs(thermalLossPct).toFixed(2)}% cold gain`;
  }

  if (csPowerEl) csPowerEl.textContent = `${csPower.toFixed(1)} kW`;
  if (ktEl) ktEl.textContent = kt.toFixed(2);
  
  if (cloudImpactEl) {
    if (kt >= 0.9) cloudImpactEl.textContent = 'Clear sky condition';
    else if (kt >= 0.6) cloudImpactEl.textContent = 'Moderate cloud attenuation';
    else cloudImpactEl.textContent = 'Heavy overcast filtering';
  }

  if (statusEl) statusEl.textContent = status;
  if (statusMetaEl) statusMetaEl.textContent = statusMeta;
}

function updateLabel(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// 4. Lightbox Modal
function initModalLightbox() {
  const modal = document.getElementById('modal-img');
  const modalImg = document.getElementById('modal-target-img');
  const modalCaption = document.getElementById('modal-caption');
  const closeBtn = document.querySelector('.modal-close');
  const backdrop = document.querySelector('.modal-backdrop');

  document.querySelectorAll('[data-modal="modal-img"]').forEach(img => {
    img.addEventListener('click', () => {
      if (modal && modalImg) {
        modalImg.src = img.src;
        if (modalCaption) modalCaption.textContent = img.alt || '';
        modal.classList.add('active');
      }
    });
  });

  const closeModal = () => {
    if (modal) modal.classList.remove('active');
  };

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  if (backdrop) backdrop.addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
}
