/**
 * Inicializa múltiplos gráficos EDA do ApexCharts.
 * @param {Object} chartsConfig - Objeto com seletores como chaves e configs como valores.
 * @param {HTMLElement} parentElement - O elemento pai onde procurar os divs dos gráficos (#eda-charts-container).
 */
async function initializeEdaCharts(chartsConfig, parentElement) {
    console.log("[initializeEdaCharts] Initializing inside:", parentElement);
    if (!parentElement) {
        console.error("[initializeEdaCharts] Invalid parentElement provided.");
        return;
    }
    try {
        await window.apexChartsLoaded;
        if (typeof ApexCharts === 'undefined') throw new Error('ApexCharts not loaded');
        console.log("[initializeEdaCharts] ApexCharts ready.");
        let delay = 0;
        let chartsFound = 0;
        for (const chartIdSelector in chartsConfig) {
            if (chartsConfig.hasOwnProperty(chartIdSelector)) {
                const chartData = chartsConfig[chartIdSelector];
                const chartDiv = parentElement.querySelector(chartIdSelector);
                if (chartDiv) {
                     chartsFound++;
                     console.log(`[initializeEdaCharts] Found ${chartIdSelector}. Queuing render with delay ${delay}ms`);
                     setTimeout((divToRender, dataForChart) => {
                          console.log(`[initializeEdaCharts] setTimeout EXECUTING for ${divToRender.id || chartIdSelector}`);
                          renderSingleChart(divToRender, dataForChart);
                     }, delay, chartDiv, chartData);
                     delay += 50;
                } else {
                     console.error(`[initializeEdaCharts] Chart div ${chartIdSelector} NOT FOUND inside parent:`, parentElement);
                }
            }
        }
         if (chartsFound > 0) {
            console.log(`✓ [initializeEdaCharts] ${chartsFound} chart renders queued.`);
        } else {
             console.error("[initializeEdaCharts] CRITICAL: No chart divs found within the parent element:", parentElement);
             if (parentElement && parentElement.innerHTML.trim().includes('spinner')) { 
                 parentElement.innerHTML = '<div class="alert alert-danger">Erro crítico: Nenhum container de gráfico EDA encontrado. Verifique seletores.</div>';
             } else if (parentElement) {
                 const errorDiv = document.createElement('div');
                 errorDiv.className = 'alert alert-danger';
                 errorDiv.textContent = 'Erro crítico: Nenhum container de gráfico EDA encontrado.';
                 parentElement.prepend(errorDiv);
             }
        }
    } catch (e) {
        console.error('[initializeEdaCharts] General failure:', e);
        const placeholders = parentElement.querySelectorAll('[id^="chart-eda-"]');
        if (placeholders.length > 0) {
             placeholders.forEach(div => {
                 div.innerHTML = '<div class="alert alert-danger small p-2 text-center">Erro: ApexCharts não disponível ou falha na inicialização.</div>';
             });
        } else if (parentElement) {
             parentElement.innerHTML = '<div class="alert alert-danger small p-2 text-center">Erro geral na inicialização dos gráficos EDA. Verifique console.</div>';
        }
    }
}
/**
 * Inicializa um único gráfico PCA do ApexCharts.
 * @param {string} chartIdSelector - O seletor CSS do div do gráfico.
 * @param {Object} chartData - O objeto de configuração do ApexCharts.
 * @param {HTMLElement} parentElement - O elemento pai onde procurar o div do gráfico (#pca-chart-container).
 */
async function initializePcaChart(chartIdSelector, chartData, parentElement) {
    console.log(`[initializePcaChart] Initializing ${chartIdSelector} inside:`, parentElement);
     if (!parentElement) {
        console.error("[initializePcaChart] Invalid parentElement provided.");
        return;
    }
    try {
        await window.apexChartsLoaded;
        if (typeof ApexCharts === 'undefined') throw new Error('ApexCharts not loaded');
        console.log("[initializePcaChart] ApexCharts ready.");
        const chartDiv = parentElement.querySelector(chartIdSelector); 
        if(chartDiv){
             console.log(`[initializePcaChart] Calling renderSingleChart for ${chartIdSelector}`);
             renderSingleChart(chartDiv, chartData);
             console.log(`✓ [initializePcaChart] Initialization complete for ${chartIdSelector}.`);
        } else {
             console.error(`[initializePcaChart] Chart div ${chartIdSelector} NOT FOUND inside parent:`, parentElement);
             parentElement.innerHTML = `<div class="alert alert-danger small p-2 text-center">Erro crítico: Div do gráfico PCA (${chartIdSelector}) não encontrado.</div>`;
        }
    } catch (e) {
        console.error(`[initializePcaChart] Failed for ${chartIdSelector}:`, e);
        const chartDiv = parentElement ? parentElement.querySelector(chartIdSelector) : null;
        if (chartDiv) chartDiv.innerHTML = '<div class="alert alert-danger small p-2 text-center">Erro no PCA. Ver console.</div>';
         else if (parentElement) parentElement.innerHTML = '<div class="alert alert-danger small p-2 text-center">Erro no PCA e div não encontrado. Ver console.</div>';
    }
}
/**
 * Função auxiliar para renderizar um gráfico ApexCharts individualmente.
 * @param {HTMLElement} chartDiv - O elemento div onde o gráfico será renderizado.
 * @param {Object} chartData - O objeto de configuração do ApexCharts.
 */
function renderSingleChart(chartDiv, chartData) {
    console.log(`[renderSingleChart] ENTERED for div:`, chartDiv);
    if (!chartDiv) { console.error(`[renderSingleChart] Invalid chartDiv provided.`); return; }
    const chartIdForLog = chartDiv.id || 'Unknown Chart Div';
    console.log(`[renderSingleChart] Rendering chart in div#${chartIdForLog}`);
    const spinnerContainer = chartDiv.querySelector('.d-flex.justify-content-center');
    if (spinnerContainer) { spinnerContainer.remove(); console.log(`[renderSingleChart] Spinner removed for ${chartIdForLog}`); }
    else { if (!chartDiv.classList.contains('apexcharts-canvas')) { chartDiv.innerHTML = ''; console.log(`[renderSingleChart] No spinner found, cleared content for ${chartIdForLog}`); } else { console.log(`[renderSingleChart] No spinner found, but div seems to contain chart already. Not clearing.`); } }
    let isValid = chartData && typeof chartData === 'object' && chartData.series && Array.isArray(chartData.series);
    if (chartData.error) { console.warn(`[renderSingleChart] Error flag present in data for ${chartIdForLog}:`, chartData.error); isValid = false; }
    if (!isValid) { console.warn(`[renderSingleChart] Invalid or error data for ${chartIdForLog}. Data:`, chartData); const errorMessage = chartData?.error || 'Gráfico indisponível ou dados inválidos.'; chartDiv.innerHTML = `<div class='alert alert-warning small p-2 text-center'>${errorMessage}</div>`; chartDiv.style.minHeight = 'auto'; return; }
    try { console.log(`[renderSingleChart] Instantiating ApexCharts for ${chartIdForLog}...`); chartDiv.style.minHeight = 'auto'; const chart = new ApexCharts(chartDiv, chartData); console.log(`[renderSingleChart] Calling chart.render() for ${chartIdForLog}...`); setTimeout(() => { chart.render().then(() => { console.log(`✓ [renderSingleChart] Chart ${chartIdForLog} rendered successfully via promise.`); }).catch(renderError => { console.error(`[renderSingleChart] Error during chart.render() promise for ${chartIdForLog}:`, renderError); chartDiv.innerHTML = '<div class="alert alert-danger small p-2 text-center">Erro durante renderização ApexCharts. Ver console.</div>'; }); }, 10); }
    catch (initError) { console.error(`[renderSingleChart] Error instantiating ApexCharts for ${chartIdForLog}:`, initError); chartDiv.innerHTML = '<div class="alert alert-danger small p-2 text-center">Erro ao iniciar ApexCharts. Ver console.</div>'; chartDiv.style.minHeight = 'auto'; }
}
document.body.addEventListener('htmx:afterSwap', function(event) {
    const swappedElement = event.detail.elt;
    console.log('[htmx:afterSwap] Event triggered. Swapped element:', swappedElement);
    function processContainer(container, type, dataAttr, chartIdSelector = null) {
        console.log(`[htmx:afterSwap] Found ${type} container:`, container);
        const chartConfigStr = container.getAttribute(dataAttr);
        if (!chartConfigStr || chartConfigStr.trim() === '{}' || chartConfigStr.trim() === '') {
            console.error(`[htmx:afterSwap] Attribute ${dataAttr} is empty or contains empty JSON on element`, container);
            container.innerHTML = `<div class="alert alert-danger">Erro: Configuração ${type} vazia.</div>`;
            return;
        }
        setTimeout(() => {
            console.log(`[htmx:afterSwap] setTimeout executing for ${type} container`);
            try {
                const chartConfig = JSON.parse(chartConfigStr);
                if (type === 'EDA') {
                    initializeEdaCharts(chartConfig, container);
                } else if (type === 'PCA' && chartIdSelector) {
                    initializePcaChart(chartIdSelector, chartConfig, container);
                }
            } catch (e) {
                console.error(`[htmx:afterSwap] Failed to parse JSON for ${type} or initialize after timeout:`, e);
                container.innerHTML = `<div class="alert alert-danger">Erro ao processar config ${type} após espera. Ver console.</div>`;
            }
        }, 10); 
    }
    if (swappedElement) {
        if (swappedElement.id === 'eda-charts-container' && swappedElement.hasAttribute('data-charts-config')) {
            processContainer(swappedElement, 'EDA', 'data-charts-config');
            return;
        }
        if (swappedElement.id === 'pca-chart-container' && swappedElement.hasAttribute('data-chart-config')) {
            processContainer(swappedElement, 'PCA', 'data-chart-config', '#chart-pca-1');
            return;
        }
        console.log('[htmx:afterSwap] Swapped element is not a primary chart container. Searching inside...');
        const edaContainerInside = swappedElement.querySelector('#eda-charts-container[data-charts-config]');
        const pcaContainerInside = swappedElement.querySelector('#pca-chart-container[data-chart-config]');
        if(edaContainerInside) {
             processContainer(edaContainerInside, 'EDA', 'data-charts-config');
        } else {
            console.log('[htmx:afterSwap] No EDA container found inside swapped element.');
        }
        if(pcaContainerInside) {
            processContainer(pcaContainerInside, 'PCA', 'data-chart-config', '#chart-pca-1');
        } else {
             console.log('[htmx:afterSwap] No PCA container found inside swapped element.');
        }
    } else {
         console.warn('[htmx:afterSwap] Swapped element (event.detail.elt) is null or undefined.');
    }
});