# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Main.py'],  # 指定主程序文件
    pathex=[os.path.dirname(os.getcwd())],  # 设置为当前工作目录的上一级目录
    binaries=[],  # 如果有额外的二进制文件，可以在这里添加
    datas=[
        # 添加需要打包的资源文件
        ('./UI\\Weld_Seam_Defect_Detection.ui', './UI'),  # UI文件
        ('./weight\\best_model_test_150.pth', './weight'),  # U-Net模型文件
        ('./weight\\best.pt', './weight'),  # YOLOv5模型文件
        ('./torch_cache', 'torch_cache'),      # 如果需要的话添加缓存目录
        ('./Predict_fun\\Predict_Single_Image.py', './Predict_fun'),  # 预测脚本
        ('./Predict_fun\\predict_yoloV5.py', './Predict_fun'),  # YOLOv5预测脚本
        ('./Predict_fun\\yolov5_thread.py', './Predict_fun'),  # YOLOv5线程脚本

    ],
    hiddenimports=['torch', 'cv2', 'numpy', 'PyQt5','ultralytics','queue'],  # 隐藏导入的库
    hookspath=[],  # 如果有自定义的hook文件，可以在这里指定
    hooksconfig={},
    runtime_hooks=[],  # 运行时hook
    excludes=[],  # 排除不需要的模块
    noarchive=False,
    optimize=0,  # 优化级别
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Main',
)
