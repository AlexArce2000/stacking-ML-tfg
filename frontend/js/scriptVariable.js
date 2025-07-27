document.addEventListener('DOMContentLoaded', function() {
    // --- CONFIGURACIÓN INICIAL ---
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
            document.getElementById('current-year').textContent = new Date().getFullYear();
        }, 800);
    }

    // --- REFERENCIAS A ELEMENTOS DEL DOM ---
    const searchInput = document.getElementById('variable-search');
    const categoryButtons = document.querySelectorAll('.category-btn');
    
    const targetWrapper = document.getElementById('target-table').closest('.table-wrapper');
    const predictorsWrapper = document.getElementById('predictors-table').closest('.table-wrapper');
    const intermediateWrapper = document.getElementById('intermediate-table').closest('.table-wrapper');
    const allRows = document.querySelectorAll('.variable-row');

    // --- FILTRO POR CATEGORÍA (LÓGICA ACTUALIZADA) ---
    categoryButtons.forEach(button => {
        button.addEventListener('click', function() {
            categoryButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            const category = this.dataset.category;

            allRows.forEach(row => row.style.display = '');

            targetWrapper.style.display = (category === 'all' || category === 'target') ? 'block' : 'none';
            predictorsWrapper.style.display = (category === 'all' || category === 'predictors') ? 'block' : 'none';
            intermediateWrapper.style.display = (category === 'all' || category === 'intermediate') ? 'block' : 'none';

            searchInput.dispatchEvent(new Event('input'));
        });
    });

    // --- FILTRO DE BÚSQUEDA ---
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase().trim();
        
        allRows.forEach(row => {
            const wrapperIsVisible = row.closest('.table-wrapper').style.display !== 'none';
            
            if (wrapperIsVisible) {
                const rowText = row.textContent.toLowerCase();
                if (rowText.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            }
        });
    });

    // --- RESALTAR FILAS AL PASAR EL MOUSE ---
    const tableRows = document.querySelectorAll('.report-table tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = 'rgba(138, 63, 252, 0.1)';
        });
        
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });

    // --- COPIAR NOMBRE DE VARIABLE AL HACER CLIC ---
    const variableCells = document.querySelectorAll('td code');
    variableCells.forEach(cell => {
        cell.addEventListener('click', function() {
            const textToCopy = this.textContent.trim();
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                this.classList.add('copied');
                setTimeout(() => {
                    this.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                console.error('Error al copiar la variable: ', err);
            });
        });
        
        cell.style.cursor = 'pointer';
        cell.title = 'Haz clic para copiar';
    });
});