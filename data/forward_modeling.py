"""
ERT正演模拟封装 - 基于pyGIMLi
提供统一的ERT正演接口，支持多种电极装置
"""

import numpy as np
import pygimli as pg
from pygimli.physics import ert


def create_electrodes(n_elec=24, spacing=1.0, x_start=None):
    """创建电极坐标

    Args:
        n_elec: 电极数量
        spacing: 电极间距 (m)
        x_start: 起始x坐标，默认居中

    Returns:
        elecs: 电极坐标数组 (n_elec, 2)  [[x1,0], [x2,0], ...]
    """
    if x_start is None:
        x_start = -(n_elec - 1) * spacing / 2
    x = np.linspace(x_start, x_start + (n_elec - 1) * spacing, n_elec)
    elecs = np.column_stack([x, np.zeros(n_elec)])
    return elecs


def create_survey_scheme(elecs, array_type='wenner'):
    """创建测量装置

    Args:
        elecs: 电极坐标数组 (n_elec, 2)
        array_type: 装置类型 'wenner', 'slumberger', 'dipole-dipole'

    Returns:
        scheme: pyGIMLi测量装置
    """
    elec_x = elecs[:, 0]

    scheme_map = {
        'wenner': 'wa',
        'schlumberger': 'slm',
        'slumberger': 'slm',  # 常见拼写变体
        'dipole-dipole': 'dd',
    }
    scheme_name = scheme_map.get(array_type, 'wa')

    scheme = ert.createData(elecs=elec_x, schemeName=scheme_name)
    return scheme


def run_forward(mesh, rho_model, scheme, noise_level=0.0):
    """运行ERT正演模拟

    Args:
        mesh: pyGIMLi网格
        rho_model: 电阻率模型 (Ω·m)，与网格单元数一致
        scheme: 测量装置
        noise_level: 噪声水平 (0.01 = 1%)

    Returns:
        data: 正演数据，包含视电阻率、电极坐标等
    """
    # 正演模拟
    data = ert.simulate(
        mesh,
        res=rho_model,
        scheme=scheme,
        verbose=False,
        noiseLevel=0  # 先不加pyGIMLi内部噪声
    )

    # 添加噪声
    if noise_level > 0:
        rhoa = data('rhoa')
        noise = np.random.normal(0, noise_level, len(rhoa))
        data.set('rhoa', rhoa * (1 + noise))
        data.set('err', np.ones(len(rhoa)) * noise_level)

    return data


def get_data_array(data):
    """从正演数据中提取视电阻率数组

    Args:
        data: pyGIMLi正演数据

    Returns:
        rhoa: 视电阻率数组
    """
    return np.array(data('rhoa'))


def get_electrode_positions(data):
    """获取电极位置

    Args:
        data: pyGIMLi正演数据

    Returns:
        elecs: 电极坐标数组 (n_elec, 2)
    """
    sensor_positions = data.sensorPositions()
    return np.array([[p.x(), p.y()] for p in sensor_positions])


def get_geometry_info(data):
    """获取数据几何信息（四点索引）

    Args:
        data: pyGIMLi正演数据

    Returns:
        geo: (n_data, 4) 数组，每行 [a, b, m, n] 电极索引
    """
    a = np.array(data('a'), dtype=int)
    b = np.array(data('b'), dtype=int)
    m = np.array(data('m'), dtype=int)
    n = np.array(data('n'), dtype=int)
    return np.column_stack([a, b, m, n])
