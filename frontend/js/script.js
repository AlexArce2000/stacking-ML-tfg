document.addEventListener('DOMContentLoaded', function() {

    const dataPath = 'frontend/data/';
    const errorOverlay = document.getElementById('error-overlay');
    let hasLoadFailed = false;

    // --- MANEJO DE ERRORES ---
    function showErrorOverlay(errorMessage) {
        if (hasLoadFailed) return;
        hasLoadFailed = true;
        console.error("Error de carga:", errorMessage);
        if (errorOverlay) errorOverlay.style.display = 'flex';
    }

    // --- CARGA DE RECURSOS ---
    function loadImage(elementId, fileName) {
        const element = document.getElementById(elementId);
        if (element) {
            element.onerror = () => showErrorOverlay(`No se pudo cargar la imagen: ${fileName}`);
            element.src = dataPath + fileName;
        }
    }
    
    loadImage('riskMapImage', 'mapa_riesgo.png');
    loadImage('performancePlot', 'performance_plot.png');
    loadImage('shapSummaryPlot', 'shap_summary_plot.png');
    loadImage('shapBarPlot', 'shap_bar_plot.png');
    loadImage('distribucionMuestra', 'mapa_distribucion_muestra.png');
    loadImage('particionEspacial', 'mapa_particion_espacial.png');

    // --- GALERÍA DE BOXPLOTS ---
    const boxplotDisplay = document.getElementById('boxplotDisplay');
    const boxplotBtns = document.querySelectorAll('.boxplot-btn');
    const initialBoxplot = document.querySelector('.boxplot-btn.active');
    if (initialBoxplot) loadImage('boxplotDisplay', initialBoxplot.dataset.src);
    
    boxplotBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            boxplotBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadImage('boxplotDisplay', btn.dataset.src);
        });
    });

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

        let html = '<div class="summary-metrics">';
        html += `<p>Accuracy: <span>${summaryMetrics['Accuracy'] || 'N/A'}</span></p>`;
        html += `<p>ROC AUC Score: <span>${summaryMetrics['ROC AUC Score'] || 'N/A'}</span></p>`;
        html += '</div>';

        html += '<table class="report-table"><thead><tr>' + tableHeader + '</tr></thead><tbody>';
tableRows.forEach(row => {
    const columns = row.trim().split(/\s+/);
    let htmlRow = '';

    if (columns[0] === 'accuracy' && columns.length === 3) {
        // accuracy 0.8440 8447
        htmlRow = `<tr>
            <td>accuracy</td>
            <td></td>
            <td></td>
            <td>${columns[1]}</td>
            <td>${columns[2]}</td>
        </tr>`;
    } else {
        const labelEndIndex = columns.length - 4;
        const label = columns.slice(0, labelEndIndex).join(' ');
        const metrics = columns.slice(labelEndIndex);
        htmlRow = `<tr><td>${label}</td>` + metrics.map(m => `<td>${m}</td>`).join('') + '</tr>';
    }

    html += htmlRow;
});

        html += '</tbody></table>';
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

        // Obtener encabezados incluso si hay solo un espacio entre ellos
        const headerMatches = [...headerLine.matchAll(/\S+/g)];
        const headerNames = headerMatches.map(m => m[0]);

        html += '<table class="report-table">';
        html += '<thead><tr><th>Índice</th>';
        headerNames.forEach(name => html += `<th>${name}</th>`);
        html += '</tr></thead><tbody>';

        dataLines.forEach(line => {
            const parts = line.trim().split(/\s{2,}/); // divide por 2 o más espacios
            const index = parts[0];
            const values = parts.slice(1);

            html += `<tr><td>${index}</td>`;
            values.forEach(cell => html += `<td>${cell}</td>`);
            html += '</tr>';
        });

        html += '</tbody></table>';
    }

    return html;
}

    
    function loadAndDisplayReport(containerId, fileName, parserFunction) {
        fetch(dataPath + fileName)
            .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.text(); })
            .then(text => { document.getElementById(containerId).innerHTML = parserFunction(text); })
            .catch(error => showErrorOverlay(`No se pudo procesar el reporte: ${fileName}. ${error.message}`));
    }

    loadAndDisplayReport('classificationReportContainer', 'classification_report.txt', parseClassificationReport);
    loadAndDisplayReport('datasetSummaryContainer', 'dataset_summary.txt', parseDatasetSummary);
    
    // --- LÓGICA DEL LIGHTBOX (ZOOM) ---
    const lightbox = document.getElementById('lightbox-overlay');
    const lightboxImage = document.getElementById('lightbox-image');
    const zoomableImages = document.querySelectorAll('.zoomable');

    zoomableImages.forEach(image => {
        image.addEventListener('click', () => {
            lightboxImage.src = image.src;
            lightbox.style.display = 'flex';
        });
    });

    lightbox.addEventListener('click', (e) => {
        if (e.target !== lightboxImage) {
            lightbox.style.display = 'none';
        }
    });

    const zoomControls = document.getElementById('zoom-controls');
const zoomInBtn = document.getElementById('zoom-in');
const zoomOutBtn = document.getElementById('zoom-out');
const zoomResetBtn = document.getElementById('zoom-reset');

let zoomLevel = 1;

zoomableImages.forEach(image => {
    image.addEventListener('click', () => {
        lightboxImage.src = image.src;
        zoomLevel = 1;
        lightboxImage.style.transform = 'scale(1)';
        lightbox.style.display = 'flex';
        zoomControls.style.display = 'flex';
    });
});

lightbox.addEventListener('click', (e) => {
    if (e.target !== lightboxImage && e.target !== zoomControls && !zoomControls.contains(e.target)) {
        lightbox.style.display = 'none';
        zoomControls.style.display = 'none';
    }
});

// Botones de zoom
zoomInBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    zoomLevel = Math.min(zoomLevel + 0.2, 5);
    lightboxImage.style.transform = `scale(${zoomLevel})`;
});

zoomOutBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    zoomLevel = Math.max(zoomLevel - 0.2, 0.5);
    lightboxImage.style.transform = `scale(${zoomLevel})`;
});

zoomResetBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    zoomLevel = 1;
    lightboxImage.style.transform = `scale(1)`;
});

});
