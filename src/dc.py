import rospy
from std_msgs.msg import Float64  

def timer_callback(message):
    print("Received timing data:", message.data)

if __name__ == '__main__':
    rospy.init_node('timer_data_listener')
    rospy.Subscriber('topic_name', Float64, timer_callback) 
    rospy.spin()
