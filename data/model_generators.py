"""
地质模型生成器
生成三种典型地质模型：污染羽扩散、CO2封存、矿体+断层
"""

import numpy as np
import pygimli as pg
from pygimli import meshtools
from scipy.spatial import cKDTree


def create_base_world(x_range=(-10, 10), depth=-10, elec_positions=None):
    """创建基础世界边界

    Args:
        x_range: x范围
        depth: 深度（负值）
        elec_positions: 电极x坐标数组（未使用，保留接口兼容）
    """
    world = meshtools.createWorld(
        start=[x_range[0], 0],
        end=[x_range[1], depth],
        worldMarker=True
    )
    return world


def create_mesh(shapes, area=0.5):
    """从几何体列表创建网格"""
    mesh = meshtools.createMesh(shapes, area=area)
    return mesh


def generate_plume_model(x_range=(-10, 10), depth=-10, seed=None, elec_x=None):
    """生成污染羽扩散模型

    低阻污染羽（10-50 Ω·m）在均匀背景（100 Ω·m）中

    Returns:
        mesh: pyGIMLi网格
        rho: 电阻率向量
        rho_matrix: 电阻率矩阵（用于可视化）
    """
    if seed is not None:
        np.random.seed(seed)

    world = create_base_world(x_range, depth, elec_x)

    # 随机生成污染羽中心和大小
    cx = np.random.uniform(x_range[0] + 3, x_range[1] - 3)
    cy = np.random.uniform(depth + 2, -2)
    rx = np.random.uniform(1.0, 3.0)
    ry = np.random.uniform(0.5, 2.0)

    # 椭圆形污染羽
    plume = meshtools.createCircle(
        pos=[cx, cy],
        radius=[rx, ry],
        marker=2,
        boundaryMarker=10
    )

    mesh = create_mesh([world, plume], area=0.3)

    # 电阻率分配
    rho = pg.Vector(mesh.cellCount())
    rho.setVal(100.0)  # 背景
    for cell in mesh.cells():
        if cell.marker() == 2:
            rho[cell.id()] = np.random.uniform(10, 50)

    # 生成可视化矩阵
    rho_matrix = _mesh_to_grid(mesh, rho, x_range, depth)

    return mesh, rho, rho_matrix


def generate_co2_model(x_range=(-10, 10), depth=-10, seed=None, elec_x=None):
    """生成CO2封存模型

    高阻CO2团块（500-2000 Ω·m）在含水层（50 Ω·m）中，上覆盖层（200 Ω·m）

    Returns:
        mesh: pyGIMLi网格
        rho: 电阻率向量
        rho_matrix: 电阻率矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    world = create_base_world(x_range, depth, elec_x)

    # 盖层
    caprock = meshtools.createRectangle(
        start=[x_range[0], -2],
        end=[x_range[1], 0],
        marker=1
    )

    # 含水层
    aquifer = meshtools.createRectangle(
        start=[x_range[0], -6],
        end=[x_range[1], -2],
        marker=2
    )

    # 基底
    basement = meshtools.createRectangle(
        start=[x_range[0], depth],
        end=[x_range[1], -6],
        marker=3
    )

    # CO2团块（1-3个）
    co2_bodies = []
    n_co2 = np.random.randint(1, 4)
    for _ in range(n_co2):
        cx = np.random.uniform(x_range[0] + 2, x_range[1] - 2)
        cy = np.random.uniform(-5, -3)
        rx = np.random.uniform(0.5, 2.0)
        ry = np.random.uniform(0.3, 1.5)
        co2 = meshtools.createCircle(
            pos=[cx, cy],
            radius=[rx, ry],
            marker=4
        )
        co2_bodies.append(co2)

    shapes = [world, caprock, aquifer, basement] + co2_bodies
    mesh = create_mesh(shapes, area=0.3)

    # 电阻率分配
    rho = pg.Vector(mesh.cellCount())
    for cell in mesh.cells():
        marker = cell.marker()
        if marker == 1:  # 盖层
            rho[cell.id()] = np.random.uniform(150, 300)
        elif marker == 2:  # 含水层
            rho[cell.id()] = np.random.uniform(30, 80)
        elif marker == 3:  # 基底
            rho[cell.id()] = np.random.uniform(200, 500)
        elif marker == 4:  # CO2
            rho[cell.id()] = np.random.uniform(500, 2000)
        else:
            rho[cell.id()] = 100.0

    rho_matrix = _mesh_to_grid(mesh, rho, x_range, depth)

    return mesh, rho, rho_matrix


def generate_fault_model(x_range=(-10, 10), depth=-10, seed=None, elec_x=None):
    """生成矿体+断层模型

    断层切割地层，矿体（低阻1-20 Ω·m）沿断层分布

    Returns:
        mesh: pyGIMLi网格
        rho: 电阻率向量
        rho_matrix: 电阻率矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    world = create_base_world(x_range, depth, elec_x)

    # 断层位置
    fault_x = np.random.uniform(x_range[0] + 3, x_range[1] - 3)
    fault_angle = np.random.uniform(60, 80)  # 断层倾角

    # 上盘地层
    upper_layer1 = meshtools.createRectangle(
        start=[x_range[0], -1.5],
        end=[x_range[1], 0],
        marker=1
    )
    upper_layer2 = meshtools.createRectangle(
        start=[x_range[0], -4],
        end=[x_range[1], -1.5],
        marker=2
    )

    # 矿体（沿断层分布的低阻体）
    ore_bodies = []
    n_ore = np.random.randint(1, 4)
    for i in range(n_ore):
        ox = fault_x + np.random.uniform(-2, 2)
        oy = np.random.uniform(-6, -2)
        ore = meshtools.createCircle(
            pos=[ox, oy],
            radius=[np.random.uniform(0.5, 1.5), np.random.uniform(0.3, 1.0)],
            marker=3
        )
        ore_bodies.append(ore)

    # 基底
    basement = meshtools.createRectangle(
        start=[x_range[0], depth],
        end=[x_range[1], -4],
        marker=4
    )

    shapes = [world, upper_layer1, upper_layer2, basement] + ore_bodies
    mesh = create_mesh(shapes, area=0.3)

    # 电阻率分配
    rho = pg.Vector(mesh.cellCount())
    for cell in mesh.cells():
        marker = cell.marker()
        if marker == 1:  # 表土层
            rho[cell.id()] = np.random.uniform(50, 150)
        elif marker == 2:  # 中间层
            rho[cell.id()] = np.random.uniform(100, 300)
        elif marker == 3:  # 矿体
            rho[cell.id()] = np.random.uniform(1, 20)
        elif marker == 4:  # 基底
            rho[cell.id()] = np.random.uniform(300, 1000)
        else:
            rho[cell.id()] = 100.0

    rho_matrix = _mesh_to_grid(mesh, rho, x_range, depth)

    return mesh, rho, rho_matrix


def generate_random_model(x_range=(-10, 10), depth=-10, seed=None, elec_x=None):
    """生成随机地质模型（用于预训练数据）

    随机组合：层状 + 块状异常体

    Returns:
        mesh: pyGIMLi网格
        rho: 电阻率向量
        rho_matrix: 电阻率矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    world = create_base_world(x_range, depth, elec_x)

    shapes = [world]

    # 随机层数 (1-4层)
    n_layers = np.random.randint(1, 5)
    layer_depths = sorted(np.random.uniform(depth + 1, -0.5, n_layers - 1))
    all_depths = [0] + list(layer_depths) + [depth]

    marker = 1
    for i in range(len(all_depths) - 1):
        layer = meshtools.createRectangle(
            start=[x_range[0], all_depths[i + 1]],
            end=[x_range[1], all_depths[i]],
            marker=marker
        )
        shapes.append(layer)
        marker += 1

    # 随机异常体 (0-5个)
    n_anomaly = np.random.randint(0, 6)
    for _ in range(n_anomaly):
        cx = np.random.uniform(x_range[0] + 1, x_range[1] - 1)
        cy = np.random.uniform(depth + 1, -1)
        r = np.random.uniform(0.3, 2.0)

        if np.random.random() > 0.5:
            anomaly = meshtools.createCircle(
                pos=[cx, cy], radius=r, marker=marker
            )
        else:
            anomaly = meshtools.createRectangle(
                start=[cx - r, cy - r * 0.6],
                end=[cx + r, cy + r * 0.6],
                marker=marker
            )
        shapes.append(anomaly)
        marker += 1

    mesh = create_mesh(shapes, area=0.5)

    # 电阻率分配（对数均匀分布）
    rho = pg.Vector(mesh.cellCount())
    base_rho = np.random.uniform(1, 3)  # log10(ρ)
    for cell in mesh.cells():
        m = cell.marker()
        if m == 0:  # 世界边界
            rho[cell.id()] = 10 ** base_rho
        elif m <= n_layers:  # 层状
            layer_rho = base_rho + np.random.uniform(-0.5, 0.5)
            rho[cell.id()] = 10 ** layer_rho
        else:  # 异常体
            anomaly_rho = base_rho + np.random.uniform(-1.5, 1.5)
            rho[cell.id()] = 10 ** anomaly_rho

    rho_matrix = _mesh_to_grid(mesh, rho, x_range, depth)

    return mesh, rho, rho_matrix


def _mesh_to_grid(mesh, rho, x_range, depth, nx=64, nz=64):
    """将网格电阻率插值到规则网格（向量化版本，使用 KDTree 加速）

    Args:
        mesh: pyGIMLi网格
        rho: 电阻率向量
        x_range: x范围
        depth: 深度（负值）
        nx, nz: 输出网格尺寸

    Returns:
        rho_matrix: (nz, nx) 电阻率矩阵
    """
    # 提取所有单元中心坐标和对应电阻率
    n_cells = mesh.cellCount()
    centers = np.zeros((n_cells, 2))
    rho_arr = np.array(rho)
    for i in range(n_cells):
        c = mesh.cell(i).center()
        centers[i, 0] = c.x()
        centers[i, 1] = c.y()

    # 构建 KDTree，批量查询最近单元
    tree = cKDTree(centers)

    x = np.linspace(x_range[0], x_range[1], nx)
    z = np.linspace(depth, 0, nz)
    X, Z = np.meshgrid(x, z)
    grid_points = np.column_stack([X.ravel(), Z.ravel()])

    _, indices = tree.query(grid_points)
    rho_matrix = rho_arr[indices].reshape(nz, nx)

    return rho_matrix
