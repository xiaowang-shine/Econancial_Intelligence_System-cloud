// script.js - 修复版本

// 全局状态
let currentPage = 0;
let totalPages = 4;
let isAnimating = false;
let currentResults = null;
let currentTaskId = null;
let wheelCooldown = false;
let wheelScrollCount = 0; // 记录连续滚动次数
const WHEEL_THRESHOLD = 3; // 需要连续滚动3次才切换页面
let lastScrollTime = 0;
const SCROLL_COOLDOWN = 1000; // 滚动冷却时间1秒

// 初始化函数
function init() {
    console.log("初始化系统...");
    initLanguage();
    setupEventListeners();
    updateScrollIndicator();
    loadThemePreferences();

    // 确保首页可见
    switchToPage(0, true);
}

// 设置事件监听器
function setupEventListeners() {
    console.log("设置事件监听器...");

    // 文件输入框点击事件 - 修复版本
    const fileInput = document.getElementById('fileInput');
    const uploadBox = document.querySelector('.upload-box');
    const uploadBtn = document.querySelector('.upload-btn');

    if (fileInput && uploadBox) {
        let fileInputCooldown = false; // 专门用于文件输入的冷却变量

        // 点击上传按钮触发文件选择
        if (uploadBtn) {
            uploadBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                if (!fileInputCooldown) {
                    fileInputCooldown = true;
                    fileInput.click();
                    setTimeout(() => {
                        fileInputCooldown = false;
                    }, 300);
                }
            });
        }

        // 点击上传框其他区域也触发文件选择（但排除按钮区域）
        uploadBox.addEventListener('click', function(e) {
            // 如果点击的是上传按钮，已经处理过了，这里跳过
            if (e.target.closest('.upload-btn')) {
                return;
            }

            if (!fileInputCooldown) {
                fileInputCooldown = true;
                fileInput.click();
                setTimeout(() => {
                    fileInputCooldown = false;
                }, 300);
            }
        });

        // 文件选择变化事件
        fileInput.addEventListener('change', function(e) {
            console.log("文件选择变化");
            handleFileSelect(e);
        });

        // 阻止文件输入框的点击事件冒泡
        fileInput.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    }

    // 开始分析按钮
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    if (startAnalysisBtn) {
        startAnalysisBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log("开始分析按钮点击");
            handleAnalysisClick();
        });
    }
    // 滚轮事件 - 添加防抖和条件判断
    document.addEventListener('wheel', handleWheel, { passive: false });

    // 触摸事件（移动端）
    let touchStartY = 0;
    document.addEventListener('touchstart', (e) => {
        touchStartY = e.touches[0].clientY;
    });

    document.addEventListener('touchend', (e) => {
        if (isAnimating) return;

        const touchEndY = e.changedTouches[0].clientY;
        const diff = touchStartY - touchEndY;

        if (Math.abs(diff) > 50) {
            const direction = diff > 0 ? 1 : -1;
            switchPage(direction);
        }
    });

    // 设置按钮
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsSidebar = document.getElementById('settingsSidebar');
    if (settingsBtn && settingsSidebar) {
        settingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            settingsSidebar.classList.toggle('open');
        });
    }

    // 点击页面其他区域关闭设置侧边栏
    document.addEventListener('click', (e) => {
        if (settingsSidebar && settingsSidebar.classList.contains('open') &&
            !settingsSidebar.contains(e.target) &&
            e.target !== settingsBtn) {
            settingsSidebar.classList.remove('open');
        }
    });

    // 首页按钮
    const homeBtn = document.getElementById('homeBtn');
    if (homeBtn) {
        homeBtn.addEventListener('click', () => {
            switchToPage(0);
        });
    }

    // 语言切换
    const languageSelector = document.getElementById('languageSelector');
    if (languageSelector) {
        languageSelector.addEventListener('change', function() {
            applyLanguage(this.value);
        });
    }

    // 主题颜色选择
    setupThemeSelectors();

    // 上传框点击事件
    setupUploadBox();
}

// 设置上传框交互
function setupUploadBox() {
    const uploadBox = document.querySelector('.upload-box');
    const fileInput = document.getElementById('fileInput');

    if (uploadBox && fileInput) {
        // 拖放功能
        uploadBox.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });

        uploadBox.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });

        uploadBox.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');

            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect({ target: fileInput });
            }
        });
    }
}

// 文件选择处理函数
function handleFileSelect(e) {
    console.log("处理文件选择");
    const file = e.target.files[0];
    if (!file) {
        console.log("没有选择文件");
        return;
    }

    // 验证文件类型
    const validTypes = ['.xls', '.xlsx', '.csv'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(fileExtension)) {
        showError('fileTypeError');
        resetFileInput();
        return;
    }

    // 显示文件信息
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileStatus = document.getElementById('fileStatus');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');

    if (fileInfo && fileName && uploadPlaceholder) {
        uploadPlaceholder.style.display = 'none';
        fileInfo.textContent = translations[currentLanguage]['selectedFile'] + ':';
        fileInfo.style.display = 'block';
        fileName.textContent = file.name;
        fileName.style.display = 'block';
    }

    if (fileStatus) {
        fileStatus.style.display = 'block';
    }

    console.log("文件选择完成:", file.name);
}

// 重置文件输入
function resetFileInput() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileStatus = document.getElementById('fileStatus');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');

    if (fileInput) fileInput.value = '';
    if (fileInfo && fileName && uploadPlaceholder) {
        fileInfo.style.display = 'none';
        fileName.style.display = 'none';
        uploadPlaceholder.style.display = 'block';
    }
    if (fileStatus) fileStatus.style.display = 'none';
}

// 设置主题选择器
function setupThemeSelectors() {
    // 色板选择
    const colorOptions = document.querySelectorAll('.color-option');
    colorOptions.forEach(option => {
        option.addEventListener('click', function() {
            // 移除其他选项的active类
            colorOptions.forEach(opt => opt.classList.remove('active'));
            // 添加active类到当前选项
            this.classList.add('active');
            // 应用主题
            const theme = this.getAttribute('data-theme');
            applyTheme(theme);
        });
    });

    // 自定义颜色选择器
    const customColorPicker = document.getElementById('customColorPicker');
    if (customColorPicker) {
        customColorPicker.addEventListener('change', function() {
            applyCustomTheme(this.value);
        });
    }
}

// 应用主题
function applyTheme(themeName) {
    // 移除所有主题类
    document.body.classList.remove('blue-theme', 'green-theme', 'purple-theme', 'orange-theme', 'dark-theme', 'custom-theme');
    // 添加新主题类
    document.body.classList.add(themeName);

    // 保存主题偏好
    localStorage.setItem('preferredTheme', themeName);

    // 如果是自定义主题，保存颜色值
    if (themeName === 'custom-theme') {
        const color = document.getElementById('customColorPicker').value;
        localStorage.setItem('customThemeColor', color);
    }
}

// 应用自定义主题
function applyCustomTheme(color) {
    // 创建自定义主题样式
    let style = document.getElementById('custom-theme-style');
    if (!style) {
        style = document.createElement('style');
        style.id = 'custom-theme-style';
        document.head.appendChild(style);
    }

    // 计算辅助颜色（稍微暗一点）
    const darkerColor = shadeColor(color, -20);

    style.textContent = `
        .custom-theme {
            --primary-color: ${color};
            --secondary-color: ${darkerColor};
        }
    `;

    // 应用自定义主题
    applyTheme('custom-theme');
}

// 颜色辅助函数：调整颜色亮度
function shadeColor(color, percent) {
    let R = parseInt(color.substring(1, 3), 16);
    let G = parseInt(color.substring(3, 5), 16);
    let B = parseInt(color.substring(5, 7), 16);

    R = parseInt(R * (100 + percent) / 100);
    G = parseInt(G * (100 + percent) / 100);
    B = parseInt(B * (100 + percent) / 100);

    R = (R < 255) ? R : 255;
    G = (G < 255) ? G : 255;
    B = (B < 255) ? B : 255;

    return "#" + ((1 << 24) + (R << 16) + (G << 8) + B).toString(16).slice(1);
}

// 加载主题偏好
function loadThemePreferences() {
    const savedTheme = localStorage.getItem('preferredTheme') || 'blue-theme';
    const customColor = localStorage.getItem('customThemeColor');

    if (savedTheme === 'custom-theme' && customColor) {
        document.getElementById('customColorPicker').value = customColor;
        applyCustomTheme(customColor);
    } else {
        applyTheme(savedTheme);

        // 激活对应的色板选项
        const activeOption = document.querySelector(`.color-option[data-theme="${savedTheme}"]`);
        if (activeOption) {
            document.querySelectorAll('.color-option').forEach(opt => opt.classList.remove('active'));
            activeOption.classList.add('active');
        }
    }
}

// 页面切换函数 - 修复动画问题
function switchPage(direction) {
    if (isAnimating || wheelCooldown) return;

    const newPage = currentPage + direction;
    if (newPage < 0 || newPage >= totalPages) return;

    isAnimating = true;
    wheelCooldown = true; // 防止在动画期间再次触发

    // 显示首页按钮（如果不是首页）
    const homeBtn = document.getElementById('homeBtn');
    if (homeBtn) {
        if (newPage > 0) {
            homeBtn.classList.add('visible');
        } else {
            homeBtn.classList.remove('visible');
        }
    }

    // 获取当前页面和新页面
    const currentPageElement = document.getElementById(`page${currentPage + 1}`);
    const newPageElement = document.getElementById(`page${newPage + 1}`);

    if (!currentPageElement || !newPageElement) {
        isAnimating = false;
        wheelCooldown = false;
        return;
    }

    // 设置动画方向类
    if (direction > 0) {
        // 向下切换页面
        currentPageElement.classList.remove('active');
        currentPageElement.classList.add('exit-up');

        newPageElement.classList.remove('enter-up');
        newPageElement.classList.add('active');
        newPageElement.classList.add('enter-down');
    } else {
        // 向上切换页面
        currentPageElement.classList.remove('active');
        currentPageElement.classList.add('exit-down');

        newPageElement.classList.remove('enter-down');
        newPageElement.classList.add('active');
        newPageElement.classList.add('enter-up');
    }

    // 如果是切换到结果页面且需要加载数据
    if (newPage > 0 && currentResults && newPageElement.querySelector('.page-content').children.length === 0) {
        loadPageContent(newPage);
    }

    // 动画结束后清理
    setTimeout(() => {
        currentPageElement.classList.remove('exit-up', 'exit-down', 'enter-up', 'enter-down');
        newPageElement.classList.remove('enter-up', 'enter-down');

        currentPage = newPage;
        isAnimating = false;
        updateScrollIndicator();

        // 动画结束后解除冷却
        setTimeout(() => {
            wheelCooldown = false;
        }, 100);
    }, 800);
}

// 直接切换到指定页面
function switchToPage(pageIndex, immediate = false) {
    if (pageIndex === currentPage || isAnimating) return;

    if (immediate) {
        // 立即切换，无动画
        const currentPageElement = document.getElementById(`page${currentPage + 1}`);
        const newPageElement = document.getElementById(`page${pageIndex + 1}`);

        if (currentPageElement) {
            currentPageElement.classList.remove('active', 'exit-up', 'exit-down', 'enter-up', 'enter-down');
        }
        if (newPageElement) {
            newPageElement.classList.add('active');
            newPageElement.classList.remove('enter-up', 'enter-down');
        }

        currentPage = pageIndex;
        updateScrollIndicator();
        return;
    }

    const direction = pageIndex > currentPage ? 1 : -1;
    const steps = Math.abs(pageIndex - currentPage);

    // 递归切换，确保动画顺序
    function switchStep(step) {
        if (step < steps) {
            setTimeout(() => {
                switchPage(direction);
                switchStep(step + 1);
            }, 400); // 减少延迟时间，让切换更流畅
        }
    }

    switchStep(0);
}

// 更新滚动指示器
function updateScrollIndicator() {
    const dots = document.querySelectorAll('.scroll-dot');
    dots.forEach((dot, index) => {
        if (index === currentPage) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
}

// 滚轮事件处理 - 修复敏感度问题
function handleWheel(e) {
    if (isAnimating || wheelCooldown) {
        e.preventDefault();
        return;
    }

    const currentTime = Date.now();
    const deltaY = e.deltaY;

    // 冷却时间检查
    if (currentTime - lastScrollTime < SCROLL_COOLDOWN) {
        e.preventDefault();
        return;
    }

    // 检查页面内容是否可以滚动
    const currentPageElement = document.getElementById(`page${currentPage + 1}`);
    if (!currentPageElement) return;

    const pageContent = currentPageElement.querySelector('.page-content');
    let shouldPreventDefault = true;

    if (pageContent && pageContent.scrollHeight > pageContent.clientHeight) {
        // 页面内容可以滚动，优先滚动内容
        const isAtTop = pageContent.scrollTop === 0;
        const isAtBottom = pageContent.scrollTop + pageContent.clientHeight >= pageContent.scrollHeight - 10;

        // 只有在内容顶部向上滚动或底部向下滚动时才切换页面
        if ((deltaY < 0 && isAtTop) || (deltaY > 0 && isAtBottom)) {
            // 允许切换页面
        } else {
            // 允许内容正常滚动，不阻止默认行为
            shouldPreventDefault = false;
        }
    }

    if (shouldPreventDefault) {
        e.preventDefault();

        // 直接切换页面，简化逻辑
        const direction = deltaY > 0 ? 1 : -1;
        switchPage(direction);
        lastScrollTime = currentTime;

        // 设置短暂的冷却时间防止连续滚动
        wheelCooldown = true;
        setTimeout(() => {
            wheelCooldown = false;
        }, 300);
    }
}

// 处理页面切换的滚轮逻辑
function handlePageSwitchWheel(deltaY) {
    const currentTime = Date.now();

    // 重置计数器（如果超过1秒没有滚动）
    if (currentTime - lastScrollTime > 1000) {
        wheelScrollCount = 0;
    }

    wheelScrollCount++;
    lastScrollTime = currentTime;

    // 只有连续滚动达到阈值才切换页面
    if (wheelScrollCount >= WHEEL_THRESHOLD) {
        const direction = deltaY > 0 ? 1 : -1;
        switchPage(direction);
        wheelScrollCount = 0; // 重置计数器
    }
}

// 分析按钮点击处理
async function handleAnalysisClick() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput || !fileInput.files[0]) {
        showError('noFileError');
        return;
    }

    try {
        setLoadingState(true);
        const results = await startAnalysis(fileInput.files[0]);
        onAnalysisSuccess(results);
    } catch (error) {
        onAnalysisError(error);
    } finally {
        setLoadingState(false);
    }
}

// 设置加载状态
function setLoadingState(isLoading) {
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    if (startAnalysisBtn) {
        if (isLoading) {
            startAnalysisBtn.textContent = translations[currentLanguage]['analyzing'] || '分析中...';
            startAnalysisBtn.disabled = true;
            startAnalysisBtn.style.opacity = '0.7';
        } else {
            startAnalysisBtn.textContent = translations[currentLanguage]['startAnalysis'];
            startAnalysisBtn.disabled = false;
            startAnalysisBtn.style.opacity = '1';
        }
    }
}

// 分析成功处理
function onAnalysisSuccess(results) {
    currentResults = results;
    showSuccess('analysisSuccess');

    // 切换到结果页面
    setTimeout(() => {
        switchToPage(1);
    }, 1000);
}

// 分析错误处理
function onAnalysisError(error) {
    console.error('分析失败:', error);
    showError('analysisError');
}

// 加载页面内容
function loadPageContent(pageIndex) {
    const pageElement = document.getElementById(`page${pageIndex + 1}`);
    if (!pageElement || !currentResults) return;

    const pageContent = pageElement.querySelector('.page-content');

    switch (pageIndex) {
        case 1: // 预测结果页面
            pageContent.innerHTML = createForecastResultsHTML();
            setTimeout(() => {
                renderForecastChart();
                renderForecastTable();
            }, 100);
            break;
        case 2: // 健康度分析页面
            pageContent.innerHTML = createHealthAnalysisHTML();
            setTimeout(() => {
                renderHealthChart();
                renderHealthTable();
            }, 100);
            break;
        case 3: // 仪表盘页面
            pageContent.innerHTML = createDashboardHTML();
            setTimeout(() => {
                renderKPIChart();
            }, 100);
            break;
    }

    // 应用语言到新内容
    applyLanguage(currentLanguage);
}

// 创建各页面的HTML模板函数
function createForecastResultsHTML() {
    return `
        <div class="results-container">
            <h2 data-i18n="financialForecast">财务预测结果</h2>
            <div class="chart-container">
                <canvas id="forecastChart"></canvas>
            </div>
            <div class="table-responsive">
                <table class="table">
                    <thead>
                        <tr>
                            <th data-i18n="date">日期</th>
                            <th data-i18n="currentAssets">流动资产</th>
                            <th data-i18n="currentLiabilities">流动负债</th>
                            <th data-i18n="inventory">存货</th>
                            <th data-i18n="totalAssets">总资产</th>
                            <th data-i18n="totalRevenue">总收入</th>
                            <th data-i18n="netProfit">净利润</th>
                            <th data-i18n="healthScore">健康度评分</th>
                        </tr>
                    </thead>
                    <tbody id="forecastTableBody"></tbody>
                </table>
            </div>
            <div class="export-buttons">
                <button class="export-btn" onclick="exportData('xlsx')" data-i18n="exportExcel">导出Excel</button>
                <button class="export-btn" onclick="exportData('csv')" data-i18n="exportCSV">导出CSV</button>
                <button class="export-btn" onclick="exportReport()" data-i18n="generatePDF">生成PDF报告</button>
            </div>
        </div>
    `;
}

function createHealthAnalysisHTML() {
    return `
        <div class="results-container">
            <h2 data-i18n="healthAnalysis">健康度分析与对比</h2>
            <div class="chart-container">
                <canvas id="healthChart"></canvas>
            </div>
            <div class="table-responsive">
                <table class="table">
                    <thead>
                        <tr>
                            <th data-i18n="date">日期</th>
                            <th data-i18n="directForecast">直接预测</th>
                            <th data-i18n="indirectForecast">间接预测</th>
                            <th data-i18n="difference">差异</th>
                            <th data-i18n="suggestions">建议</th>
                        </tr>
                    </thead>
                    <tbody id="healthTableBody"></tbody>
                </table>
            </div>
            <div id="healthSuggestions" class="suggestions-container"></div>
        </div>
    `;
}

function createDashboardHTML() {
    return `
        <div class="results-container">
            <h2 data-i18n="dashboard">KPI仪表盘</h2>
            <div class="chart-container">
                <canvas id="kpiChart"></canvas>
            </div>
        </div>
    `;
}

// 图表渲染函数
function renderForecastChart() {
    const ctx = document.getElementById('forecastChart');
    if (!ctx || !currentResults) return;

    const forecastData = currentResults.financial_forecast?.data || currentResults.results?.financial_forecast?.data || [];
    if (forecastData.length === 0) return;

    const labels = forecastData.map(item => item.date);
    const healthScores = forecastData.map(item => item.health_score || 0);
    const revenues = forecastData.map(item => item.Rev || item.revenue || 0);

    new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '健康度评分',
                    data: healthScores,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: true,
                    yAxisID: 'y',
                    tension: 0.4
                },
                {
                    label: '营业收入',
                    data: revenues,
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    fill: true,
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: '健康度评分'
                    },
                    min: 0,
                    max: 1
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '营业收入'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '未来12个月财务健康度预测'
                }
            }
        }
    });
}

function renderForecastTable() {
    const tbody = document.getElementById('forecastTableBody');
    if (!tbody || !currentResults) return;

    const forecastData = currentResults.financial_forecast?.data || currentResults.results?.financial_forecast?.data || [];

    if (forecastData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center">暂无数据</td></tr>';
        return;
    }

    tbody.innerHTML = forecastData.map(item => `
        <tr>
            <td>${item.date}</td>
            <td>${formatNumber(item.CA)}</td>
            <td>${formatNumber(item.CL)}</td>
            <td>${formatNumber(item.Inv)}</td>
            <td>${formatNumber(item.TA)}</td>
            <td>${formatNumber(item.Rev)}</td>
            <td>${formatNumber(item.NI)}</td>
            <td>
                <div class="health-score">
                    <div class="score-value">${((item.health_score || 0) * 100).toFixed(1)}%</div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${(item.health_score || 0) * 100}%"></div>
                    </div>
                </div>
            </td>
        </tr>
    `).join('');
}

function renderHealthChart() {
    const ctx = document.getElementById('healthChart');
    if (!ctx || !currentResults) return;

    const healthData = currentResults.health_comparison?.data || currentResults.results?.health_comparison?.data || [];
    if (healthData.length === 0) return;

    const labels = healthData.map(item => item.date);
    const directForecasts = healthData.map(item => item.direct_forecast || 0);
    const indirectForecasts = healthData.map(item => item.indirect_forecast || 0);

    new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '直接预测',
                    data: directForecasts,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    fill: false,
                    tension: 0.4
                },
                {
                    label: '间接预测',
                    data: indirectForecasts,
                    borderColor: '#95a5a6',
                    backgroundColor: 'rgba(149, 165, 166, 0.1)',
                    fill: false,
                    tension: 0.4,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: '健康度评分'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '健康度预测对比'
                }
            }
        }
    });
}

function renderHealthTable() {
    const tbody = document.getElementById('healthTableBody');
    if (!tbody || !currentResults) return;

    const healthData = currentResults.health_comparison?.data || currentResults.results?.health_comparison?.data || [];
    if (healthData.length === 0) return;

    tbody.innerHTML = healthData.map(item => {
        const direct = item.direct_forecast || 0;
        const indirect = item.indirect_forecast || 0;
        const diff = direct - indirect;
        const diffPercent = indirect !== 0 ? (diff / indirect * 100).toFixed(1) : '0.0';

        let suggestion = '';
        if (diff > 0.1) {
            suggestion = '乐观：直接预测明显高于间接预测';
        } else if (diff < -0.1) {
            suggestion = '谨慎：直接预测低于间接预测';
        } else {
            suggestion = '稳定：预测结果一致';
        }

        return `
            <tr>
                <td>${item.date}</td>
                <td>${(direct * 100).toFixed(1)}%</td>
                <td>${(indirect * 100).toFixed(1)}%</td>
                <td class="${diff >= 0 ? 'text-success' : 'text-danger'}">
                    ${diff >= 0 ? '+' : ''}${diffPercent}%
                </td>
                <td>${suggestion}</td>
            </tr>
        `;
    }).join('');
}

function renderKPIChart() {
    const ctx = document.getElementById('kpiChart');
    if (!ctx || !currentResults) return;

    // 简单的KPI图表实现
    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['健康度评分', '营业收入', '净利润', '流动比率', '速动比率'],
            datasets: [{
                label: '关键指标',
                data: [0.85, 100000, 20000, 2.13, 1.64], // 示例数据
                backgroundColor: ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '关键绩效指标'
                }
            }
        }
    });
}

// 数字格式化函数
function formatNumber(num) {
    if (num === undefined || num === null) return '0';
    if (num === 0) return '0';

    if (Math.abs(num) >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (Math.abs(num) >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    } else {
        return num.toFixed(2);
    }
}

// 分析函数
async function startAnalysis(file) {
    try {
        setLoadingState(true);

        const formData = new FormData();
        formData.append('files', file);

        const response = await fetch('/analyze_financials', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP错误! 状态码: ${response.status}`);
        }

        const responseText = await response.text();
        console.log('原始响应:', responseText);

        let result;
        try {
            result = JSON.parse(responseText);
        } catch (jsonError) {
            console.error('JSON解析错误:', jsonError);
            const cleanedText = responseText
                .replace(/NaN/g, 'null')
                .replace(/Infinity/g, 'null')
                .replace(/-Infinity/g, 'null');

            try {
                result = JSON.parse(cleanedText);
            } catch (secondError) {
                throw new Error(`无效的JSON响应: ${responseText.substring(0, 200)}...`);
            }
        }

        if (result.status === 'success') {
            currentTaskId = result.task_id;
            const cleanedResults = cleanNaNValues(result.results);
            return cleanedResults;
        } else {
            throw new Error(result.message || '分析失败');
        }

    } catch (error) {
        console.error('分析错误:', error);
        showError('分析失败: ' + error.message);
        throw error;
    }
}

// 清理数据中的 NaN 值
function cleanNaNValues(data) {
    if (typeof data !== 'object' || data === null) {
        return data;
    }

    if (Array.isArray(data)) {
        return data.map(item => cleanNaNValues(item));
    }

    const cleaned = {};
    for (const [key, value] of Object.entries(data)) {
        if (typeof value === 'number' && (isNaN(value) || !isFinite(value))) {
            cleaned[key] = 0;
        } else if (typeof value === 'object' && value !== null) {
            cleaned[key] = cleanNaNValues(value);
        } else {
            cleaned[key] = value;
        }
    }
    return cleaned;
}

// 错误和成功提示函数
function showError(messageKey) {
    const message = translations[currentLanguage][messageKey] || messageKey;

    // 移除现有的错误提示
    const existingError = document.querySelector('.error-message');
    if (existingError) existingError.remove();

    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #e74c3c;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        z-index: 10000;
        max-width: 400px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        animation: slideInRight 0.3s ease;
    `;

    errorDiv.innerHTML = `
        <strong>${translations[currentLanguage]['error'] || '错误'}:</strong> ${message}
        <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; cursor: pointer; color: white; font-size: 16px;">×</button>
    `;

    document.body.appendChild(errorDiv);

    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

function showSuccess(messageKey) {
    const message = translations[currentLanguage][messageKey] || messageKey;

    // 移除现有的成功提示
    const existingSuccess = document.querySelector('.success-message');
    if (existingSuccess) existingSuccess.remove();

    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #2ecc71;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        z-index: 10000;
        max-width: 400px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        animation: slideInRight 0.3s ease;
    `;

    successDiv.innerHTML = `
        <strong>${translations[currentLanguage]['success'] || '成功'}:</strong> ${message}
        <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; cursor: pointer; color: white; font-size: 16px;">×</button>
    `;

    document.body.appendChild(successDiv);

    setTimeout(() => {
        if (successDiv.parentElement) {
            successDiv.remove();
        }
    }, 3000);
}

// 导出功能
async function exportData(format) {
    if (!currentTaskId) {
        showError('没有可导出的任务');
        return;
    }

    try {
        const response = await fetch(`/export_data?task_id=${currentTaskId}&format=${format}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `financial_forecast.${format}`;
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            showError('导出失败');
        }
    } catch (error) {
        console.error('导出失败:', error);
        showError('导出失败');
    }
}

async function exportReport() {
    if (!currentTaskId) {
        showError('没有可导出的报告');
        return;
    }

    try {
        const response = await fetch(`/export_report?task_id=${currentTaskId}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'financial_report.pdf';
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            showError('生成报告失败');
        }
    } catch (error) {
        console.error('生成报告失败:', error);
        showError('生成报告失败');
    }
}

// 语言切换功能
function initLanguage() {
    const savedLanguage = localStorage.getItem('preferredLanguage') || 'zh';
    applyLanguage(savedLanguage);

    // 设置语言选择器的值
    const languageSelector = document.getElementById('languageSelector');
    if (languageSelector) {
        languageSelector.value = savedLanguage;
    }
}

function applyLanguage(lang) {
    currentLanguage = lang;

    // 更新所有带有 data-i18n 属性的元素
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (translations[lang] && translations[lang][key]) {
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                element.placeholder = translations[lang][key];
            } else {
                element.textContent = translations[lang][key];
            }
        }
    });

    // 更新页面标题
    document.title = translations[lang]['appName'];

    // 保存语言设置到本地存储
    localStorage.setItem('preferredLanguage', lang);
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* 添加向下退出和向上进入的动画 */
    .page.exit-down {
        animation: slideDownExit 0.8s ease-out forwards;
    }
    
    .page.enter-down {
        animation: slideDownEnter 0.8s ease-out forwards;
    }
    
    @keyframes slideDownExit {
        from {
            transform: translateY(0);
            opacity: 1;
        }
        to {
            transform: translateY(100vh);
            opacity: 0;
        }
    }
    
    @keyframes slideDownEnter {
        from {
            transform: translateY(-100vh);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .text-success { color: #2ecc71; font-weight: 600; }
    .text-danger { color: #e74c3c; font-weight: 600; }
    .text-center { text-align: center; }
    
    /* 确保页面内容可以滚动 */
    .page-content {
        max-height: calc(100vh - 100px);
        overflow-y: auto;
        padding: 20px;
    }
    
    /* 隐藏滚动条但保持滚动功能 */
    .page-content::-webkit-scrollbar {
        width: 5px;
    }
    
    .page-content::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
        border-radius: 10px;
    }
    
    .page-content::-webkit-scrollbar-thumb {
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
    }
`;
document.head.appendChild(style);

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM加载完成，开始初始化...");
    init();
});