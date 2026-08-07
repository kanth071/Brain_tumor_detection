document.addEventListener('DOMContentLoaded', () => {
    // Nav Items & Views
    const navAnalysis = document.getElementById('navAnalysis');
    const navReports = document.getElementById('navReports');
    const navSettings = document.getElementById('navSettings');

    const viewAnalysis = document.getElementById('viewAnalysis');
    const viewReports = document.getElementById('viewReports');
    const viewSettings = document.getElementById('viewSettings');

    const navItems = [navAnalysis, navReports, navSettings];
    const views = [viewAnalysis, viewReports, viewSettings];

    // Analysis View Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileTag = document.getElementById('fileTag');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const btnRemoveFile = document.getElementById('btnRemoveFile');
    const btnRunAnalysis = document.getElementById('btnRunAnalysis');
    const btnIcon = document.getElementById('btnIcon');
    const btnText = document.getElementById('btnText');

    const systemStatusPill = document.getElementById('systemStatusPill');
    const systemStatusText = document.getElementById('systemStatusText');

    const previewEmptyState = document.getElementById('previewEmptyState');
    const scanPreviewImg = document.getElementById('scanPreviewImg');
    const previewFilenameTag = document.getElementById('previewFilenameTag');

    const diagnosticEmptyState = document.getElementById('diagnosticEmptyState');
    const diagnosticResults = document.getElementById('diagnosticResults');
    const gaugeFill = document.getElementById('gaugeFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const predictionClass = document.getElementById('predictionClass');

    const gradcamEmptyState = document.getElementById('gradcamEmptyState');
    const gradcamImg = document.getElementById('gradcamImg');
    const layerTag = document.getElementById('layerTag');

    // Reports View Elements
    const reportsTableBody = document.getElementById('reportsTableBody');
    const btnRefreshReports = document.getElementById('btnRefreshReports');
    const reportSearchInput = document.getElementById('reportSearchInput');
    const statTotalScans = document.getElementById('statTotalScans');
    const statTumorCount = document.getElementById('statTumorCount');
    const statNonTumorCount = document.getElementById('statNonTumorCount');
    const statAvgConfidence = document.getElementById('statAvgConfidence');

    // Settings View Elements
    const settingTargetLayer = document.getElementById('settingTargetLayer');
    const settingColormap = document.getElementById('settingColormap');
    const settingAlpha = document.getElementById('settingAlpha');
    const alphaValueDisplay = document.getElementById('alphaValueDisplay');
    const settingAutoReload = document.getElementById('settingAutoReload');
    const btnForceReload = document.getElementById('btnForceReload');
    const btnSaveSettings = document.getElementById('btnSaveSettings');

    // Modal Elements
    const reportModal = document.getElementById('reportModal');
    const btnCloseModal = document.getElementById('btnCloseModal');
    const modalBody = document.getElementById('modalBody');

    let currentFile = null;
    let allReports = [];

    // ==========================================
    // 1. NAVIGATION TAB SWITCHING
    // ==========================================
    function switchTab(targetNav, targetView) {
        navItems.forEach(item => item.classList.remove('active'));
        views.forEach(view => view.classList.remove('active'));

        targetNav.classList.add('active');
        targetView.classList.add('active');

        if (targetView === viewReports) {
            loadReports();
        } else if (targetView === viewSettings) {
            loadSettings();
        }
    }

    navAnalysis.addEventListener('click', (e) => { e.preventDefault(); switchTab(navAnalysis, viewAnalysis); });
    navReports.addEventListener('click', (e) => { e.preventDefault(); switchTab(navReports, viewReports); });
    navSettings.addEventListener('click', (e) => { e.preventDefault(); switchTab(navSettings, viewSettings); });

    // ==========================================
    // 2. BACKEND STATUS CHECK
    // ==========================================
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            if (data.model_loaded) {
                console.log(`[NeuroScan AI] Model loaded: ${data.model_filename} (Layer: ${data.target_layer})`);
                syncSettingsUI(data);
            } else {
                updateSystemStatus('Model Load Error', 'error');
            }
        })
        .catch(err => console.error('[NeuroScan AI] Status check failed:', err));

    function updateSystemStatus(text, state = 'ready') {
        systemStatusText.textContent = text;
        systemStatusPill.className = 'status-pill ' + state;
    }

    // ==========================================
    // 3. FILE INPUT & ANALYSIS
    // ==========================================
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#btnRemoveFile')) return;
        fileInput.value = '';
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file) return;

        currentFile = file;
        fileNameDisplay.textContent = file.name;
        fileTag.classList.remove('hidden');
        dropZone.querySelector('.drop-text-content').classList.add('hidden');
        dropZone.querySelector('.drop-icon-wrapper').classList.add('hidden');

        const reader = new FileReader();
        reader.onload = (e) => {
            scanPreviewImg.src = e.target.result;
            scanPreviewImg.classList.remove('hidden');
            previewEmptyState.classList.add('hidden');
            previewFilenameTag.textContent = file.name;
            previewFilenameTag.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

        btnRunAnalysis.disabled = false;
        updateSystemStatus('Scan Loaded', 'ready');
    }

    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        fileTag.classList.add('hidden');
        dropZone.querySelector('.drop-text-content').classList.remove('hidden');
        dropZone.querySelector('.drop-icon-wrapper').classList.remove('hidden');

        scanPreviewImg.src = '';
        scanPreviewImg.classList.add('hidden');
        previewEmptyState.classList.remove('hidden');
        previewFilenameTag.classList.add('hidden');

        gradcamImg.src = '';
        gradcamImg.classList.add('hidden');
        gradcamEmptyState.classList.remove('hidden');
        layerTag.classList.add('hidden');

        diagnosticResults.classList.add('hidden');
        diagnosticEmptyState.classList.remove('hidden');

        btnRunAnalysis.disabled = true;
        updateSystemStatus('System Ready', 'ready');
    });

    btnRunAnalysis.addEventListener('click', () => {
        if (!currentFile) return;

        btnRunAnalysis.disabled = true;
        btnIcon.className = 'fa-solid fa-spinner fa-spin';
        btnText.textContent = 'Analyzing...';
        updateSystemStatus('Analyzing MRI Scan...', 'analyzing');

        const formData = new FormData();
        formData.append('file', currentFile);

        fetch('/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                renderResults(data);
                updateSystemStatus('Analysis Complete', 'ready');
            } else {
                alert('Analysis failed: ' + (data.error || 'Unknown error'));
                updateSystemStatus('Analysis Error', 'error');
            }
        })
        .catch(err => {
            console.error('API Error:', err);
            alert('Server connection error. Please try again.');
            updateSystemStatus('Server Error', 'error');
        })
        .finally(() => {
            btnRunAnalysis.disabled = false;
            btnIcon.className = 'fa-regular fa-circle-play';
            btnText.textContent = 'Run Analysis';
        });
    });

    function isTumorClass(prediction) {
        if (!prediction) return false;
        const lower = prediction.toLowerCase();
        return lower.includes('tumorous') && !lower.includes('non');
    }

    function renderResults(data) {
        diagnosticEmptyState.classList.add('hidden');
        diagnosticResults.classList.remove('hidden');

        const confidence = data.confidence;
        predictionClass.textContent = data.prediction;

        if (isTumorClass(data.prediction)) {
            predictionClass.style.color = '#ef4444';
            gaugeFill.style.stroke = '#ef4444';
        } else {
            predictionClass.style.color = '#10b981';
            gaugeFill.style.stroke = '#10b981';
        }

        animateCounter(confidenceValue, 0, confidence, 1200);

        const circumference = 427;
        const offset = circumference - (circumference * confidence / 100);
        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = offset;
        }, 100);

        gradcamEmptyState.classList.add('hidden');
        gradcamImg.src = data.gradcam_image;
        gradcamImg.classList.remove('hidden');
        layerTag.textContent = data.target_layer;
        layerTag.classList.remove('hidden');
    }

    function animateCounter(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = (progress * (end - start) + start).toFixed(1);
            element.textContent = `${value}%`;
            if (progress < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
    }

    // ==========================================
    // 4. REPORTS LOG & STATS
    // ==========================================
    function loadReports() {
        fetch('/api/reports')
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    allReports = data.reports;
                    renderReportsTable(allReports);
                    updateReportStats(allReports);
                }
            })
            .catch(err => console.error('Error loading reports:', err));
    }

    btnRefreshReports.addEventListener('click', loadReports);

    reportSearchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = allReports.filter(r => 
            r.report_id.toLowerCase().includes(query) ||
            r.filename.toLowerCase().includes(query) ||
            r.prediction.toLowerCase().includes(query)
        );
        renderReportsTable(filtered);
    });

    function renderReportsTable(reports) {
        if (reports.length === 0) {
            reportsTableBody.innerHTML = `
                <tr class="empty-table-row">
                    <td colspan="7">No analysis reports logged yet. Run a scan from the Analysis tab!</td>
                </tr>`;
            return;
        }

        reportsTableBody.innerHTML = reports.map(r => {
            const isTumor = isTumorClass(r.prediction);
            const badgeClass = isTumor ? 'badge-tag tumor' : 'badge-tag non-tumor';
            return `
                <tr>
                    <td><strong>${r.report_id}</strong></td>
                    <td>${r.filename}</td>
                    <td>${r.timestamp}</td>
                    <td><span class="${badgeClass}">${r.prediction}</span></td>
                    <td><strong>${r.confidence}%</strong></td>
                    <td><code>${r.target_layer}</code></td>
                    <td>
                        <button class="btn-secondary btn-sm" onclick="viewReportDetails('${r.report_id}')">
                            <i class="fa-solid fa-eye"></i> View Heatmap
                        </button>
                    </td>
                </tr>`;
        }).join('');
    }

    function updateReportStats(reports) {
        const total = reports.length;
        let tumorCount = 0;
        let nonTumorCount = 0;
        let totalConfidence = 0;

        reports.forEach(r => {
            if (isTumorClass(r.prediction)) tumorCount++;
            else nonTumorCount++;
            totalConfidence += r.confidence;
        });

        const avgConfidence = total > 0 ? (totalConfidence / total).toFixed(1) : '0';

        statTotalScans.textContent = total;
        statTumorCount.textContent = tumorCount;
        statNonTumorCount.textContent = nonTumorCount;
        statAvgConfidence.textContent = `${avgConfidence}%`;
    }

    // Modal View Report Details
    window.viewReportDetails = function(reportId) {
        const r = allReports.find(item => item.report_id === reportId);
        if (!r) return;

        modalBody.innerHTML = `
            <div class="modal-img-col">
                <span>Original MRI Scan (${r.filename})</span>
                <img src="${r.original_image}" alt="Original Scan">
            </div>
            <div class="modal-img-col">
                <span>Grad-CAM Heatmap (${r.target_layer})</span>
                <img src="${r.gradcam_image}" alt="Grad-CAM Overlay">
            </div>
        `;
        reportModal.classList.remove('hidden');
    };

    btnCloseModal.addEventListener('click', () => reportModal.classList.add('hidden'));
    reportModal.addEventListener('click', (e) => {
        if (e.target === reportModal) reportModal.classList.add('hidden');
    });

    // ==========================================
    // 5. SETTINGS CONTROLLER
    // ==========================================
    settingAlpha.addEventListener('input', (e) => {
        alphaValueDisplay.textContent = e.target.value;
    });

    function loadSettings() {
        fetch('/api/settings')
            .then(res => res.json())
            .then(data => {
                if (data.success) syncSettingsUI(data.config);
            });
    }

    function syncSettingsUI(config) {
        if (config.target_layer) settingTargetLayer.value = config.target_layer;
        if (config.colormap) settingColormap.value = config.colormap;
        if (config.alpha) {
            settingAlpha.value = config.alpha;
            alphaValueDisplay.textContent = config.alpha;
        }
        if (config.auto_reload !== undefined) settingAutoReload.checked = config.auto_reload;
    }

    btnSaveSettings.addEventListener('click', () => {
        const updatedConfig = {
            target_layer: settingTargetLayer.value,
            colormap: settingColormap.value,
            alpha: parseFloat(settingAlpha.value),
            auto_reload: settingAutoReload.checked
        };

        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedConfig)
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert('Settings saved successfully!');
            }
        })
        .catch(err => console.error('Failed to save settings:', err));
    });

    btnForceReload.addEventListener('click', () => {
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ force_reload: true })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert('Weights forced reloaded from model.h5 successfully!');
            }
        });
    });
});
