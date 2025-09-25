// languages.js - 多语言支持
const translations = {
    zh: {
        // 导航和基础
        'appName': '智能经营分析系统',
        'home': '首页',
        'settings': '设置',
        'upload': '上传',
        'analysis': '分析',
        'export': '导出',

        // 上传页面
        'uploadPlaceholder': '请上传.xls/.xlsx/.csv文件',
        'selectedFile': '已选择文件',
        'startAnalysis': '开始财务分析',
        'systemTitle': '智能经营分析系统',
        'systemDescription': '上传您的财务数据文件，系统将为您生成专业的预测分析报告',
        'supportedFormats': '支持 .xls, .xlsx, .csv 格式的文件',

        // 分析结果
        'financialForecast': '财务预测结果',
        'healthAnalysis': '健康度分析与对比',
        'dashboard': 'KPI仪表盘',
        'date': '日期',
        'forecastValue': '预测值',
        'lowerLimit': '下限',
        'upperLimit': '上限',
        'suggestions': '优化建议',

        // 设置
        'language': '语言',
        'blue': '蓝色',
        'dark': '暗色',
        'chinese': '中文',
        'english': 'English',

        // 导出相关
        'exportExcel': '导出Excel',
        'exportCSV': '导出CSV',
        'generatePDF': '生成PDF报告',

        // 表格标题
        'currentAssets': '流动资产',
        'currentLiabilities': '流动负债',
        'inventory': '存货',
        'totalAssets': '总资产',
        'totalRevenue': '总收入',
        'netProfit': '净利润',
        'healthScore': '健康度评分',
        'directForecast': '直接预测',
        'indirectForecast': '间接预测',
        'difference': '差异',

        // 错误和成功消息
        'uploadSuccess': '文件上传成功！',
        'analysisSuccess': '分析完成！',
        'analysisError': '分析失败',
        'fileTypeError': '请上传 .xls, .xlsx 或 .csv 格式的文件',
        'noFileError': '请先选择文件',

        'themeColor': '主题颜色',
        'customColor': '自定义颜色',
        'scrollHint': '滚动切换',
        'analyzing': '分析中...',
        'futureForecast': '未来12个月预测结果'
    },
    en: {
        // 导航和基础
        'appName': 'Econancial Intelligence System (EIS)',
        'home': 'Home',
        'settings': 'Settings',
        'upload': 'Upload',
        'analysis': 'Analysis',
        'export': 'Export',

        // 上传页面
        'uploadPlaceholder': 'Please upload .xls/.xlsx/.csv files',
        'selectedFile': 'Selected file',
        'startAnalysis': 'Start Financial Analysis',
        'systemTitle': 'Econancial Intelligence System',
        'systemDescription': 'Upload your financial data file, the system will generate professional forecast analysis reports',
        'supportedFormats': 'Supports .xls, .xlsx, .csv format files',

        // 分析结果
        'financialForecast': 'Financial Forecast Results',
        'healthAnalysis': 'Health Analysis and Comparison',
        'dashboard': 'KPI Dashboard',
        'date': 'Date',
        'forecastValue': 'Forecast Value',
        'lowerLimit': 'Lower Limit',
        'upperLimit': 'Upper Limit',
        'suggestions': 'Optimization Suggestions',

        // 设置
        'themeColor': 'Theme Color',
        'language': 'Language',
        'blue': 'Blue',
        'dark': 'Dark',
        'chinese': '中文',
        'english': 'English',

        // 导出相关
        'exportExcel': 'Export Excel',
        'exportCSV': 'Export CSV',
        'generatePDF': 'Generate PDF Report',

        // 表格标题
        'currentAssets': 'Current Assets',
        'currentLiabilities': 'Current Liabilities',
        'inventory': 'Inventory',
        'totalAssets': 'Total Assets',
        'totalRevenue': 'Total Revenue',
        'netProfit': 'Net Profit',
        'healthScore': 'Health Score',
        'directForecast': 'Direct Forecast',
        'indirectForecast': 'Indirect Forecast',
        'difference': 'Difference',

        // 错误和成功消息
        'uploadSuccess': 'File uploaded successfully!',
        'analysisSuccess': 'Analysis completed!',
        'analysisError': 'Analysis failed',
        'fileTypeError': 'Please upload .xls, .xlsx or .csv format files',
        'noFileError': 'Please select a file first',

        'customColor': 'Custom Color',
        'scrollHint': 'Scroll to navigate',
        'analyzing': 'Analyzing...',
        'futureForecast': 'Next 12 Months Forecast'


    }
};

// 当前语言状态
let currentLanguage = 'zh';

// 应用语言设置
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

// 初始化语言设置
function initLanguage() {
    const savedLanguage = localStorage.getItem('preferredLanguage') || 'zh';
    applyLanguage(savedLanguage);

    // 设置语言选择器的值
    const languageSelector = document.getElementById('languageSelector');
    if (languageSelector) {
        languageSelector.value = savedLanguage;
    }
}