document.addEventListener('DOMContentLoaded', function() {
    // Configuración de rutas
    const dataPath = 'frontend/output/';
    const errorOverlay = document.getElementById('error-overlay');
    const loadingOverlay = document.getElementById('loading-overlay');
    let hasLoadFailed = false;
    let loadedResources = 0;
    const totalResources = 8; // Imágenes + archivos de texto

    // Mostrar pantalla de carga inicial
    loadingOverlay.style.display = 'flex';
    
    // --- MANEJO DE ERRORES ---
    function showErrorOverlay(errorMessage) {
        if (hasLoadFailed) return;
        hasLoadFailed = true;
        console.error("Error de carga:", errorMessage);
        
        // Ocultar pantalla de carga
        loadingOverlay.style.display = 'none';
        
        // Mostrar overlay de error
        if (errorOverlay) {
            errorOverlay.style.display = 'flex';
            
            // Configurar botón de reintento
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

    // --- CARGA DE RECURSOS ---
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
            // Todas las imágenes cargadas, ocultar loader
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
                document.body.style.overflow = 'auto';
                
                // Actualizar año en el footer
                document.getElementById('current-year').textContent = new Date().getFullYear();
                
                // Actualizar fecha de última actualización
                document.getElementById('last-updated').textContent = new Date().toLocaleString();
            }, 500);
        }
    }

    // --- INICIALIZACIÓN DEL DASHBOARD ---
    function initializeDashboard() {
        // Cargar imágenes principales
        loadImage('riskMapImage', 'mapa_riesgo.png');
        loadImage('performancePlot', 'performance_plot.png');
        loadImage('shapSummaryPlot', 'shap_summary_plot.png');
        loadImage('shapBarPlot', 'shap_bar_plot.png');
        loadImage('distribucionMuestra', 'mapa_distribucion_muestra.png');
        loadImage('particionEspacial', 'mapa_particion_espacial.png');

        // Inicializar galería de boxplots
        initializeBoxplotGallery();
        
        // Cargar reportes de texto
        loadAndDisplayReport('classificationReportContainer', 'classification_report.txt', parseClassificationReport);
        loadAndDisplayReport('datasetSummaryContainer', 'dataset_summary.txt', parseDatasetSummary);
        
        // Inicializar lightbox
        initializeLightbox();
    }

    // --- GALERÍA DE BOXPLOTS ---
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

    // --- PARSEO Y VISUALIZACIÓN DE REPORTES ---
    
    // Parser para el reporte de clasificación (ya es robusto)
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

    // Mostrar métricas resumen
    let html = '<div class="summary-metrics">';
    html += `<p>Accuracy: <span>${summaryMetrics['Accuracy'] || 'N/A'}</span></p>`;
    html += `<p>ROC AUC Score: <span>${summaryMetrics['ROC AUC Score'] || 'N/A'}</span></p>`;
    html += '</div>';

    // Mostrar tabla
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

    // Actualizar KPIs visuales
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
            
            // Extraer área cubierta si está disponible
            const areaMatch = infoPart.match(/Total samples:\s*(\d+)/);
            if (areaMatch && areaMatch[1]) {
                const areaKm2 = Math.round(areaMatch[1] * 0.03); // Aproximación de 0.03 km² por muestra
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

    // --- LIGHTBOX CON ZOOM Y ARRASTRE ---
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
        }, 300); // Espera a que termine la animación
        document.body.style.overflow = '';
    }

lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) {
        closeLightbox(); // cerrar si clic fuera de imagen
    }
});

if (lightboxClose) {
    lightboxClose.addEventListener('click', (e) => {
        e.stopPropagation(); // evitar que se propague al overlay
        closeLightbox();
    });
}

    // Eventos de teclado
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

    // Controles de zoom
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

    // Arrastre de imagen
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

    // Soporte táctil para móviles
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

    // Inicializar cursor
    lightboxImage.style.cursor = 'zoom-in';
}

    // --- INICIAR LA APLICACIÓN ---
    initializeDashboard();
});