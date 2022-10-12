import tensorflow as tf
import timeit

# with tf.device('/cpu:0'):
#     cpu_a = tf.random.normal([10000, 1000])
#     cpu_b = tf.random.normal([1000, 2000])
#     # print(cpu_a.device, cpu_b.device)
#
# with tf.device('/gpu:0'):
#     gpu_a = tf.random.normal([10000, 1000])
#     gpu_b = tf.random.normal([1000, 2000])
#     print(gpu_a.device, gpu_b.device)


# def cpu_run():
#     with tf.device('/cpu:0'):
#         c = tf.matmul(cpu_a, cpu_b)
#     return c
#
#
# def gpu_run():
#     with tf.device('/gpu:0'):
#         c = tf.matmul(gpu_a, gpu_b)
#     return c


# cpu_time = timeit.timeit(cpu_run, number=10)
# gpu_time = timeit.timeit(gpu_run, number=10)
# print('warmup:', cpu_time, gpu_time)
#
# cpu_time = timeit.timeit(cpu_run, number=10)
# gpu_time = timeit.timeit(gpu_run, number=10)
# print('run time', cpu_time, gpu_time)


x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)

with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    # y = a ** 2 * x + b * x + c
    y = x ** 2

# [dy_da, dy_db, dy_dc] = tape.gradient(y, [a, b, c])
# print(dy_da, dy_db, dy_dc)

print(tape.gradient(y, x))
