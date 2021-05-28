import tensorflow as tf

state = tf.Variable( 0 ,dtype=tf.int8,name= 'MID_VAL')
one = tf.constant( 1 ,dtype=tf.int8,name='ONE')

new_val = tf.add(state, one,name='ADD')   #state+1→ new_val
update = tf.assign(state, new_val,name='update')   #update功能，更新参数
update1 = tf.assign(state, 2*one,name='update1')
def test():
    print(131)
    return update
def test1():
    print(2445)
    return update1

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(test()))
    # print(sess.run(test1()))
    # writer = tf.summary.FileWriter('.', sess.graph)
    print(sess.run([test(), test1()]))
    # print(sess.run([test(),test1()]))test
# writer.close()