import matplotlib
import logging
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import argparse
from datetime import datetime
import matplotlib.animation as animation


matplotlib.use('tkagg')
logger = logging.getLogger(__name__)
estimated_pis = []


# Set up the plot
main_fig, main_ax = plt.subplots()
line, = main_ax.plot([], [], '*', color='red')
hline = main_ax.axhline(y=0, color='coral', linestyle='--', label='Average Y')

def init():
    main_ax.axhline(y=np.pi, color='brown', label='3.14')
    main_ax.set_title("Buffon's Experiment Plot")
    main_ax.legend()
    return main_ax,

def update_fig(frame):
    y_data = estimated_pis[-50:]  # Show only last 50 points
    x_data = list(range(len(y_data)))

    line.set_data(x_data, y_data)

    if len(y_data):
        avg_y = np.mean(y_data)
        hline.set_ydata([avg_y])
    else:
        hline.set_ydata([0])

    # Adjust limits
    main_ax.relim()
    main_ax.autoscale_view()

    return line, hline


def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="Buffon's experiment plot")
    parser.add_argument('--output_file', required=False,
                        default=None, type=str,
                        help="path to the output file, which will be created if it doesn't exist")

    args = parser.parse_args()
    output_file = args.output_file
    if output_file is None:
        output_file = f"./{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"

    logger.log(logging.INFO, parser.description)
    logger.log(logging.INFO, f"Saving values to {output_file}")


    def mqtt_on_message(client, userdata, message):
        logger.log(logging.INFO, f"Received message '{message.payload.decode()}'")
        match message.topic:
            case "buffon pi":
                pi_value = float(message.payload.decode())
                estimated_pis.append(pi_value)
                with open(output_file, 'a') as file:
                    pi_value = str(pi_value) + '\n'
                    file.write(pi_value)

    mqtt_plot_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_plot_client.on_message = mqtt_on_message
    mqtt_plot_client.connect("test.mosquitto.org", 1883, 60)
    mqtt_plot_client.subscribe("buffon pi")
    mqtt_plot_client.loop_start()

    ani = animation.FuncAnimation(main_fig, update_fig,
                                  init_func=init, interval=100,
                                  cache_frame_data=False)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()