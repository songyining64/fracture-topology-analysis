# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # 主布局 (垂直)
        self.main_vbox = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_vbox.setContentsMargins(10, 10, 10, 10)

        # ==========================================
        # 上部：控制按键区 (分为三排逻辑清晰的工具栏)
        # ==========================================
        self.top_frame = QtWidgets.QFrame(self.centralwidget)
        self.top_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.top_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 5px; }")
        self.top_layout = QtWidgets.QVBoxLayout(self.top_frame)

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
            "background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 13px; padding: 5px;")

        # --- 右下侧：图片内嵌大画板 ---
        self.canvas_container = QtWidgets.QWidget(self.splitter)
        self.canvas_container.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        # 设置左右比例 (左侧输出占一小半，右侧图片占一大半)
        self.splitter.setSizes([350, 1050])
        self.main_vbox.addWidget(self.splitter, 1)  # 1代表占据剩余全部高度

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)