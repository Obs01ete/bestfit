# Нахождение прямой, проходящей через набор точек
# Дмитрий Хизбуллин, 2021

import math
import numpy as np
from typing import Optional, Any
import matplotlib.pyplot as plt


def generate_points(axis: Optional[Any] = None) -> np.ndarray:
    """
    Функция, генерирующая массив точек, приблизительно лежащих на
    отрезке прямой.

    :param axis: Набор осей, на которых рисовать график
    :return: Numpy массив формы [N, 2] точек на прямой
    """

    # Давайте сгенерируем 10 точек, лежащих на прямой
    num_points = 10
    p_initial = np.array((-2, 3))
    speed = 1.0
    angle_degrees = -30
    angle_radians = angle_degrees / 180 * math.pi
    velocity = speed * np.array((math.cos(angle_radians),
                                 math.sin(angle_radians)))
    ideal_points = np.expand_dims(p_initial, 0) + \
                   np.outer(np.arange(0, num_points), velocity)
    # и добавим немного шума, чтобы промоделировать реальные измерения
    noise = 0.1 * np.random.randn(num_points, 2)
    points = ideal_points + noise

    if axis is not None:
        for ax in axis:
            ax.set_title("Входной набор точек")
            ax.plot(points[:, 0], points[:, 1], 'or')
            ax.grid(True, linestyle='--')
            ax.axis('equal')

    return points


def least_squares(points: np.ndarray, axis: Optional[Any] = None) \
        -> np.ndarray:
    """
    Функция для аппроксимации массива точек прямой, основанная на
    методе наименьших квадратов.

    :param points: Входной массив точек формы [N, 2]
    :param axis: Набор осей, на которых рисовать график
    :return: Numpy массив формы [N, 2] точек на прямой
    """

    x = points[:, 0]
    y = points[:, 1]
    # Для метода наименьших квадратов нам нужно, чтобы X был матрицей,
    # в которой первый столбей - единицы, а второй - x координаты точек
    X = np.vstack((np.ones(x.shape[0]), x)).T
    normal_matrix = np.dot(X.T, X)
    moment_matrix = np.dot(X.T, y)
    # beta_hat это вектор [перехват, наклон], рассчитываем его в
    # в соответствии с формулой.
    beta_hat = np.dot(np.linalg.inv(normal_matrix), moment_matrix)
    intercept = beta_hat[0]
    slope = beta_hat[1]
    # Теперь, когда мы знаем параметры прямой, мы можем
    # легко вычислить y координаты точек на прямой.
    y_hat = intercept + slope * x
    # Соберем x и y в единую матрицу, которую мы собираемся вернуть
    # в качестве результата.
    points_hat = np.vstack((x, y_hat)).T

    if axis is not None:
        for ax in axis:
            ax.set_title("Метод наименьших квадратов")
            ax.plot(x, y, 'or')
            ax.plot(x, y_hat, 'o-', mfc='none')
            ax.grid(True, linestyle='--')
            ax.axis('equal')

    return points_hat


def ransac(points: np.ndarray,
           min_inliers: int = 4,
           max_distance: float = 0.15,
           outliers_fraction: float = 0.5,
           probability_of_success: float = 0.99,
           axis: Optional[Any] = None) -> Optional[np.ndarray]:
    """
    RANdom SAmple Consensus метод нахождения наилучшей
    аппроксимирующей прямой.

    :param points: Входой массив точек формы [N, 2]
    :param min_inliers: Минимальное количество не-выбросов
    :param max_distance: максимальное расстояние до поддерживающей прямой,
                         чтобы точка считалась не-выбросом
    :param outliers_fraction: Ожидаемая доля выбросов
    :param probability_of_success: желаемая вероятность, что поддерживающая
                                   прямая не основана на точке-выбросе
    :param axis: Набор осей, на которых рисовать график
    :return: Numpy массив формы [N, 2] точек на прямой,
             None, если ответ не найден.
    """

    # Давайте вычислим необходимое количество итераций
    num_trials = int(math.log(1 - probability_of_success) /
                     math.log(1 - outliers_fraction**2))

    best_num_inliers = 0
    best_support = None
    for _ in range(num_trials):
        # В каждой итерации случайным образом выбираем две точки
        # из входного массива и называем их "суппорт"
        random_indices = np.random.choice(
            np.arange(0, len(points)), size=(2,), replace=False)
        assert random_indices[0] != random_indices[1]
        support = np.take(points, random_indices, axis=0)

        # Здесь мы считаем расстояния от всех точек до прямой
        # заданной суппортом. Для расчета расстояний от точки до
        # прямой подходит функция векторного произведения.
        # Особенность np.cross в том, что функция возвращает только
        # z координату векторного произведения, а она-то нам и нужна.
        cross_prod = np.cross(support[1, :] - support[0, :],
                              support[1, :] - points)
        support_length = np.linalg.norm(support[1, :] - support[0, :])
        # cross_prod содержит знаковое расстояние, поэтому нам нужно
        # взять модуль значений.
        distances = np.abs(cross_prod) / support_length

        # Не-выбросы - это все точки, которые ближе, чем max_distance
        # к нашей прямой-кандидату.
        num_inliers = np.sum(distances < max_distance)
        # Здесь мы обновляем лучший найденный суппорт
        if num_inliers >= min_inliers and num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_support = support

    # Если мы успешно нашли хотя бы один суппорт,
    # удовлетворяющий всем требованиям
    if best_support is not None:
        # Спроецируем точки из входного массива на найденную прямую
        support_start = best_support[0]
        support_vec = best_support[1] - best_support[0]
        # Для расчета проекций отлично подходит функция
        # скалярного произведения.
        offsets = np.dot(support_vec, (points - support_start).T)
        proj_vectors = np.outer(support_vec, offsets).T
        support_sq_len = np.inner(support_vec, support_vec)
        projected_vectors = proj_vectors / support_sq_len
        projected_points = support_start + projected_vectors

        if axis is not None:
            for ax in axis:
                ax.set_title("RANSAC")
                ax.scatter(best_support[:, 0], best_support[:, 1],
                            s=200, facecolors='none', edgecolors='k', marker='s')
                ax.plot(points[:, 0], points[:, 1], 'or')
                ax.plot(projected_points[:, 0], projected_points[:, 1],
                         'o-', mfc='none')
                ax.grid(True, linestyle='--')
                ax.axis('equal')
    else:
        projected_points = None

    return projected_points


def pca(points: np.ndarray, axis: Optional[Any] = None) -> np.ndarray:
    """

    Метод главных компонент (PCA) оценки направления
    максимальной досперсии облака точек.

    :param points: Входой массив точек формы [N, 2]
    :param axis: Набор осей, на которых рисовать график
    :return: Numpy массив формы [N, 2] точек на прямой
    """

    # Найдем главные компоненты.
    # В первую очередь нужно центрировать облако точек, вычтя среднее
    mean = np.mean(points, axis=0)
    centered = points - mean
    # Функция вычисления собственных значений и векторов np.linalg.eig
    # требует ковариационную матрицу в качестве аргумента.
    cov = np.cov(centered.T)
    # Теперь мы можем посчитать главные компоненты, заданные
    # собственными значениями и собственными векторами.
    eigenval, eigenvec = np.linalg.eig(cov)
    # Мы хотим параметризовать целевую прямую в координатной системе,
    # заданной собственным вектором, собственное значение которого
    # наиболее велико (направление наибольшей вариативности).
    argmax_eigen = np.argmax(eigenval)
    # Нам понадобятся проекции входных точек на наибольший собственный
    # вектор.
    loc_pca = np.dot(centered, eigenvec)
    loc_maxeigen = loc_pca[:, argmax_eigen]
    max_eigenval = eigenval[argmax_eigen]
    max_eigenvec = eigenvec[:, argmax_eigen]
    # Ре-параметризуем прямую, взяв за начало отрезка проекции
    # первой и последней точки на прямую.
    loc_start = mean + max_eigenvec * loc_maxeigen[0]
    loc_final = mean + max_eigenvec * loc_maxeigen[-1]
    linspace = np.linspace(0, 1, num=len(points))
    # Получаем позиции точек, которые идут с одинаковым интервалом,
    # таким образом удаляя шум измерений и вдоль траектории движения.
    positions = loc_start + np.outer(linspace, loc_final - loc_start)

    if axis is not None:
        for ax in axis:
            ax.set_title("PCA")
            ax.plot(points[:, 0], points[:, 1], 'or')
            ax.plot(positions[:, 0], positions[:, 1], 'o-', mfc='none')
            ax.grid(True, linestyle='--')
            ax.axis('equal')

    return positions


def main():
    fig_all, axs = plt.subplots(2, 2)
    fig_all.set_size_inches(10.0, 6.0)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    points = generate_points(axis=(axs[0, 0], ax))
    fig.savefig('1_points.png', dpi=300)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    least_squares(points, axis=(axs[0, 1], ax))
    fig.savefig('2_leastsq.png', dpi=300)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    ransac(points, axis=(axs[1, 0], ax))
    fig.savefig('3_ransac.png', dpi=300)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    pca(points, axis=(axs[1, 1], ax))
    fig.savefig('4_pca.png', dpi=300)

    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')

    for ax in axs.flat:
        ax.label_outer()

    fig_all.savefig('0_all.png', dpi=300)

    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
