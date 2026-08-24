概率校准
========

``hscredit.core.models.calibration`` 提供二分类概率校准算法、预训练模型包装器、校准报告和中文可靠性图。
底层算法接受一维正类概率；``ProbabilityCalibrator`` 接受模型特征并保持 ``classes_`` 的真实概率列顺序。

.. autoclass:: hscredit.core.models.calibration.BaseCalibrator
   :members:

.. autoclass:: hscredit.core.models.calibration.PlattCalibrator
   :members:

.. autoclass:: hscredit.core.models.calibration.IsotonicCalibrator
   :members:

.. autoclass:: hscredit.core.models.calibration.BetaCalibrator
   :members:

.. autoclass:: hscredit.core.models.calibration.HistogramCalibrator
   :members:

.. autoclass:: hscredit.core.models.calibration.ProbabilityCalibrator
   :members:

.. autoclass:: hscredit.core.models.calibration.CalibratedModel
   :members:

.. autofunction:: hscredit.core.models.calibration.calibrate_model

.. autofunction:: hscredit.core.models.calibration.plot_calibration_comparison
