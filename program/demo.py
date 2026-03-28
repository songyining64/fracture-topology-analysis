# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 900)
        MainWindow.setWindowTitle("地质断裂拓扑智能分析系统 v1.0")
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # 主布局 (垂直)
        self.main_vbox = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_vbox.setContentsMargins(10, 10, 10, 10)

        # ==========================================
        # 上部：控制按键区 (你原来的三排排版)
        # ==========================================
        self.top_frame = QtWidgets.QFrame(self.centralwidget)
        self.top_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.top_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 5px; }")
        self.top_layout = QtWidgets.QVBoxLayout(self.top_frame)

        # --- 第零排：数据源切换 ---
        self.row0_layout = QtWidgets.QHBoxLayout()
        self.lbl_data_source = QtWidgets.QLabel("数据源：")
        self.combo_data_source = QtWidgets.QComboBox()
        self.combo_data_source.addItems([
            "准噶尔盆地车莫古隆起 (THK)",
            "柯坪断隆KB11",
            "塔里木盆地英买2 (MY)",
        ])
        self.combo_data_source.setMinimumWidth(160)
        self.combo_data_source.setMinimumHeight(35)
        self.combo_data_source.setToolTip("切换断裂迹线与研究区数据，融合分析将使用对应区域的网格CSV")

        # 通用下拉框样式（确保下拉列表背景白色、文字深色，避免系统主题导致白字白底看不见）
        _combo_style = """
            QComboBox {
                background-color: #ffffff;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 13px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #2c3e50;
                selection-background-color: #3498db;
                selection-color: #ffffff;
                border: 1px solid #bdc3c7;
                outline: none;
            }
        """
        # 仅数据源下拉略放大字号（框内与弹出列表），其余下拉仍用 _combo_style
        _combo_style_data_source = _combo_style.replace(
            "font-size: 13px;",
            "font-size: 15px;",
        ).replace(
            "QComboBox QAbstractItemView {\n                background-color:",
            "QComboBox QAbstractItemView {\n                font-size: 15px;\n                background-color:",
        )
        self.combo_data_source.setStyleSheet(_combo_style_data_source)
        self.row0_layout.addWidget(self.lbl_data_source)
        self.row0_layout.addWidget(self.combo_data_source)
        self.row0_layout.addStretch()
        self.top_layout.addLayout(self.row0_layout)

        # --- 第一排：基础地质与拓扑绘图 ---
        self.row1_layout = QtWidgets.QHBoxLayout()
        self.btn_yuantu = QtWidgets.QPushButton("原断裂数据地图")
        self.btn_fenleihou = QtWidgets.QPushButton("分类后数据地图")
        self.btn_relitu = QtWidgets.QPushButton("断裂密度热力图")
        self.btn_azimuth = QtWidgets.QPushButton("方位角集图")
        self.btn_meiguitu = QtWidgets.QPushButton("方向玫瑰图")
        self.btn_sanyuantu = QtWidgets.QPushButton("各类别三元图")
        self.btn_guanxi = QtWidgets.QPushButton("交叉与相邻关系")
        self.btn_b = QtWidgets.QPushButton("确定分支和节点")
        self.btn_a = QtWidgets.QPushButton("长度分布拟合")

        for btn in [self.btn_yuantu, self.btn_fenleihou, self.btn_relitu, self.btn_azimuth,
                    self.btn_meiguitu, self.btn_sanyuantu, self.btn_guanxi, self.btn_b, self.btn_a]:
            btn.setMinimumHeight(35)
            self.row1_layout.addWidget(btn)
        self.top_layout.addLayout(self.row1_layout)

        # --- 第二排：视图与参数提取 ---
        self.row2_layout = QtWidgets.QHBoxLayout()
        self.combo_topo = QtWidgets.QComboBox()
        self.combo_topo.addItems(["请选择拓扑视图...", "拓扑化后断裂数据地图1", "拓扑化后断裂数据地图2"])
        self.combo_params = QtWidgets.QComboBox()
        self.combo_params.addItems(["下拉选择绘制参数...", "Fracture Intensity B21 / P21 (断裂强度)",
                                    "Trace Max/Min/Mean Length (迹线长度分布)", "Dimensionless Intensity (无量纲强度)",
                                    "Number of Traces (实际迹线数量)", "Branch Max/Min/Mean Length", "Node Count",
                                    "Branch Count"])
        self.btn_tuopushuxing = QtWidgets.QPushButton("显示拓扑属性数据")

        self.combo_topo.setStyleSheet(_combo_style)
        self.combo_params.setStyleSheet(_combo_style)
        for widget in [self.combo_topo, self.combo_params, self.btn_tuopushuxing]:
            widget.setMinimumHeight(35)
            self.row2_layout.addWidget(widget)
        self.top_layout.addLayout(self.row2_layout)

        # --- 第三排：机器学习与属性融合 ---
        self.row3_layout = QtWidgets.QHBoxLayout()
        self.combo_fusion = QtWidgets.QComboBox()
        self.combo_fusion.addItems(["融合方法: PCA", "融合方法: 自编码器", "融合方法: UMAP", "融合方法: VAE"])
        self.btn_ronghe = QtWidgets.QPushButton("执行属性融合分析")
        self.btn_guoji_weighted = QtWidgets.QPushButton("高价值属性加权融合")
        self.btn_guoji_compare = QtWidgets.QPushButton("融合对比(加权vsGAT)")
        self.btn_guoji_train = QtWidgets.QPushButton("训练XGBoost模型")
        self.btn_guoji_shap = QtWidgets.QPushButton("SHAP可解释分析")
        self.btn_spatial = QtWidgets.QPushButton("一键空间-拓扑融合")

        self.combo_fusion.setStyleSheet(_combo_style)
        for widget in [self.combo_fusion, self.btn_ronghe, self.btn_guoji_weighted, self.btn_guoji_compare,
                       self.btn_guoji_train, self.btn_guoji_shap, self.btn_spatial]:
            widget.setMinimumHeight(35)
            self.row3_layout.addWidget(widget)
        self.top_layout.addLayout(self.row3_layout)

        self.main_vbox.addWidget(self.top_frame)

        # ==========================================
        # 下部：左右分栏结构
        # ==========================================
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # --- 左侧：信息与日志输出区 ---
        self.text_browser = QtWidgets.QTextBrowser(self.splitter)
        self.text_browser.setMinimumWidth(350)
        self.text_browser.setStyleSheet(
            "background-color: #f8f9fa; "
            "color: #2c3e50; "
            "border: 1px solid #dcdde1; "
            "border-radius: 4px; "
            "font-family: Consolas, 'Courier New', monospace; "
            "font-size: 17px; "
            "padding: 8px; "
            "line-height: 1.5;"
        )

        # --- 右下侧：图片内嵌大画板 ---
        self.canvas_container = QtWidgets.QWidget(self.splitter)
        self.canvas_container.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter.setSizes([350, 1050])
        self.main_vbox.addWidget(self.splitter, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)