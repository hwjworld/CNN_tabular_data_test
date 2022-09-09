import numpy as np


def compute_loss_for_line_given_points(b, w, points):
    totalLoss = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalLoss += (y - (w * x + b)) ** 2
    return totalLoss / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = w_current - (learning_rate * w_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate,
                            num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt("data_randint.csv", delimiter=",")
    learning_rate = 0.0003
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 500000
    print("Starting gradient descent at b={0},w={1}, loss={2}".format(
        initial_b, initial_w, compute_loss_for_line_given_points(initial_b,
                                                                  initial_w,
                                                                  points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w,
                                     learning_rate, num_iterations)
    print("After {0} iterations b={1}, w={2}, loss={3}".format(
        num_iterations, b, w,
        compute_loss_for_line_given_points(b, w, points)))


if __name__ == '__main__':
    # 用学习解 方程 y=wx+b
    run()
    # print("")
    # for i in range(100):
    #     with open("data_randint.csv", "a") as f:
    #         inn = random.randint(0,100)
    #         f.write("{},{}\n".format(inn,inn*2+random.randint(0,10)))