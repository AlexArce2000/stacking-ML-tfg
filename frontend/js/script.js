document.addEventListener('DOMContentLoaded', function() {
    const dataPath = 'frontend/output/';
    const errorOverlay = document.getElementById('error-overlay');
    const loadingOverlay = document.getElementById('loading-overlay');
    let hasLoadFailed = false;
    let loadedResources = 0;
    const totalResources = 8; 

    loadingOverlay.style.display = 'flex';
    
    function showErrorOverlay(errorMessage) {
        if (hasLoadFailed) return;
        hasLoadFailed = true;
        console.error("Error de carga:", errorMessage);
        
        loadingOverlay.style.display = 'none';
        
        if (errorOverlay) {
            errorOverlay.style.display = 'flex';
            
            const retryButton = document.getElementById('retry-button');
            if (retryButton) {
                retryButton.addEventListener('click', function() {
                    errorOverlay.style.display = 'none';
                    hasLoadFailed = false;
                    loadedResources = 0;
                    loadingOverlay.style.display = 'flex';
                    initializeDashboard();
                });
            }
        }
    }

    function loadImage(elementId, fileName) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const img = new Image();
        img.onload = function() {
            element.src = dataPath + fileName;
            resourceLoaded();
        };
        img.onerror = function() {
            showErrorOverlay(`No se pudo cargar la imagen: ${fileName}`);
        };
        img.src = dataPath + fileName;
    }
    
    function resourceLoaded() {
        loadedResources++;
        if (loadedResources === totalResources) {
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
                document.body.style.overflow = 'auto';
                
                document.getElementById('current-year').textContent = new Date().getFullYear();
                
                document.getElementById('last-updated').textContent = new Date().toLocaleString();
            }, 500);
        }
    }

    function initializeDashboard() {
        loadImage('riskMapImage', 'risk_map_heatmap_final.png');
        loadImage('performancePlot', 'performance_plot.png');
        loadImage('shapSummaryPlot', 'shap_summary_plot_rf.png');
        loadImage('shapBarPlot', 'shap_bar_plot_rf.png');
        loadImage('distribucionMuestra', 'mapa_distribucion_muestra.png');
        loadImage('particionEspacial', 'mapa_particion_espacial.png');

        initializeBoxplotGallery();
        loadAndDisplayReport('classificationReportContainer', 'classification_report.txt', parseClassificationReport);
        loadAndDisplayReport('datasetSummaryContainer', 'dataset_summary.txt', parseDatasetSummary);
        initializeLightbox();
    }

    function initializeBoxplotGallery() {
        const boxplotDisplay = document.getElementById('boxplotDisplay');
        const boxplotBtns = document.querySelectorAll('.boxplot-btn');
        const initialBoxplot = document.querySelector('.boxplot-btn.active');
        
        if (initialBoxplot && boxplotDisplay) {
            loadImage('boxplotDisplay', initialBoxplot.dataset.src);
        }
        
        boxplotBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                boxplotBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                loadImage('boxplotDisplay', btn.dataset.src);
            });
        });
    }
function parseClassificationReport(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const summaryMetrics = {};
    const tableRows = [];
    let tableHeader = '';

    lines.forEach(line => {
        const trimmedLine = line.trim();
        if (trimmedLine.startsWith('Accuracy:')) {
            summaryMetrics['Accuracy'] = trimmedLine.split(/:\s*/)[1];
        } else if (trimmedLine.startsWith('ROC AUC Score:')) {
            summaryMetrics['ROC AUC Score'] = trimmedLine.split(/:\s*/)[1];
        } else if (trimmedLine.startsWith('precision')) {
            tableHeader = `<th>Clase</th>` + trimmedLine.split(/\s+/).map(h => `<th>${h}</th>`).join('');
        } else if (trimmedLine.match(/^(No Incendio|Incendio|accuracy|macro avg|weighted avg)/)) {
            tableRows.push(trimmedLine);
        }
    });

    let html = '<div class="summary-metrics">';
    html += `<p>Accuracy: <span>${summaryMetrics['Accuracy'] || 'N/A'}</span></p>`;
    html += `<p>ROC AUC Score: <span>${summaryMetrics['ROC AUC Score'] || 'N/A'}</span></p>`;
    html += '</div>';

    html += '<table class="report-table"><thead><tr>' + tableHeader + '</tr></thead><tbody>';
    
    tableRows.forEach(row => {
        const columns = row.trim().split(/\s+/);
        let htmlRow = '';

        if (columns[0] === 'accuracy' && columns.length === 3) {
            htmlRow = `<tr class="highlight">
                <td>accuracy</td>
                <td colspan="2"></td>
                <td>${columns[1]}</td>
                <td>${columns[2]}</td>
            </tr>`;
        } else {
            const labelEndIndex = columns.length - 4;
            const label = columns.slice(0, labelEndIndex).join(' ');
            const metrics = columns.slice(labelEndIndex);
            
            htmlRow = `<tr><td>${label}</td>` + 
                metrics.map(m => `<td>${m}</td>`).join('') + 
                '</tr>';
        }

        html += htmlRow;
    });

    html += '</tbody></table>';

    const fireRow = tableRows.find(row => row.includes('Incendio'));
    if (fireRow) {
        const fireCount = fireRow.split(/\s+/).pop();
        const fireCountElement = document.getElementById('fire-count');
        if (fireCountElement) {
            fireCountElement.textContent = fireCount;
        }
    }

    const accuracyElement = document.getElementById('accuracy');
    if (accuracyElement) {
        const rawAcc = summaryMetrics['Accuracy'];
        const asPercent = parseFloat(rawAcc) * 100;
        accuracyElement.textContent = isNaN(asPercent) ? rawAcc : `${asPercent.toFixed(1)}%`;
    }

    resourceLoaded();
    return html;
}

    function parseDatasetSummary(text) {
        const parts = text.split('--- DataFrame .head() ---');
        const infoPart = parts[0].replace(/---.*?---/g, '').trim();
        const headPart = parts[1] ? parts[1].trim() : '';

        let html = `<div class="dataset-info-block"><pre>${infoPart}</pre></div>`;

        if (headPart) {
            const lines = headPart.split('\n').filter(l => l.trim());
            const headerLine = lines[0];
            const dataLines = lines.slice(1);

            // Obtener encabezados
            const headerMatches = [...headerLine.matchAll(/\S+/g)];
            const headerNames = headerMatches.map(m => m[0]);

            html += '<table class="report-table">';
            html += '<thead><tr><th>Índice</th>';
            headerNames.forEach(name => html += `<th>${name}</th>`);
            html += '</tr></thead><tbody>';

            dataLines.forEach(line => {
                const parts = line.trim().split(/\s{2,}/);
                const index = parts[0];
                const values = parts.slice(1);

                html += `<tr><td>${index}</td>`;
                values.forEach(cell => html += `<td>${cell}</td>`);
                html += '</tr>';
            });

            html += '</tbody></table>';
            
            const areaMatch = infoPart.match(/Total samples:\s*(\d+)/);
            if (areaMatch && areaMatch[1]) {
                const areaKm2 = Math.round(areaMatch[1] * 0.03); 
                const areaElement = document.getElementById('area-covered');
                if (areaElement) {
                    areaElement.textContent = areaKm2.toLocaleString();
                }
            }
        }
        
        resourceLoaded();
        return html;
    }

    function loadAndDisplayReport(containerId, fileName, parserFunction) {
        fetch(dataPath + fileName)
            .then(response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.text();
            })
            .then(text => {
                const container = document.getElementById(containerId);
                if (container) {
                    container.innerHTML = parserFunction(text);
                }
            })
            .catch(error => {
                showErrorOverlay(`No se pudo procesar el reporte: ${fileName}. ${error.message}`);
            });
    }

function initializeLightbox() {
    const lightbox = document.getElementById('lightbox-overlay');
    const lightboxImage = document.getElementById('lightbox-image');
    const zoomableImages = document.querySelectorAll('.zoomable');
    const zoomInBtn = document.getElementById('zoom-in');
    const zoomOutBtn = document.getElementById('zoom-out');
    const zoomResetBtn = document.getElementById('zoom-reset');
    const lightboxClose = document.querySelector('.lightbox-close');

    let isDragging = false;
    let startX = 0, startY = 0;
    let translateX = 0, translateY = 0;
    let scale = 1;

    function updateImageTransform() {
        lightboxImage.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    }

    function resetZoom() {
        scale = 1;
        translateX = 0;
        translateY = 0;
        updateImageTransform();
        lightboxImage.style.cursor = 'zoom-in';
    }

    // Abrir lightbox al hacer clic en imágenes
    zoomableImages.forEach(image => {
        image.addEventListener('click', () => {
            lightboxImage.src = image.src;
            resetZoom();
            lightbox.style.display = 'flex';
            setTimeout(() => {
                lightbox.classList.add('show');
            }, 10);
            document.body.style.overflow = 'hidden';
        });
    });

    function closeLightbox() {
        lightbox.classList.remove('show');
        setTimeout(() => {
            lightbox.style.display = 'none';
        }, 300); 
        document.body.style.overflow = '';
    }

lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) {
        closeLightbox(); 
    }
});

if (lightboxClose) {
    lightboxClose.addEventListener('click', (e) => {
        e.stopPropagation(); 
        closeLightbox();
    });
}

    document.addEventListener('keydown', (e) => {
        if (!lightbox.classList.contains('show')) return;
        
        switch(e.key) {
            case 'Escape':
                closeLightbox();
                break;
            case '+':
            case '=':
                if (e.ctrlKey || e.metaKey) {
                    scale = Math.min(scale + 0.2, 5);
                    updateImageTransform();
                }
                break;
            case '-':
                if (e.ctrlKey || e.metaKey) {
                    scale = Math.max(scale - 0.2, 0.5);
                    updateImageTransform();
                }
                break;
            case '0':
                resetZoom();
                break;
        }
    });

    zoomInBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        scale = Math.min(scale + 0.2, 5);
        updateImageTransform();
        lightboxImage.style.cursor = 'grab';
    });

    zoomOutBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        scale = Math.max(scale - 0.2, 0.5);
        updateImageTransform();
        if (scale <= 1) {
            resetZoom();
        }
    });

    zoomResetBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetZoom();
    });

    lightboxImage.addEventListener('mousedown', (e) => {
        if (scale <= 1) return;
        isDragging = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
        lightboxImage.style.cursor = 'grabbing';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        updateImageTransform();
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        if (scale > 1) {
            lightboxImage.style.cursor = 'grab';
        } else {
            lightboxImage.style.cursor = 'zoom-in';
        }
    });

    lightboxImage.addEventListener('touchstart', (e) => {
        if (scale <= 1) return;
        isDragging = true;
        const touch = e.touches[0];
        startX = touch.clientX - translateX;
        startY = touch.clientY - translateY;
        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        const touch = e.touches[0];
        translateX = touch.clientX - startX;
        translateY = touch.clientY - startY;
        updateImageTransform();
        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchend', () => {
        isDragging = false;
    });

    lightboxImage.style.cursor = 'zoom-in';
}

    initializeDashboard();
});