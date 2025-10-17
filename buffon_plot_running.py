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
num_needles_list = []
num_crossed_list = []
estimated_pis_list = []
cummulative_pis_list = []

# Set up the plot
main_fig, main_ax = plt.subplots()
estimated_pis, = main_ax.plot([], [], 'x', color='red', label='individual exps')
cummulative_pis, = main_ax.plot([], [], '*-', color='coral', label='cummulative')
#hline = main_ax.axhline(y=0, color='coral', linestyle='--', label='Average Y')

def init():
    main_ax.axhline(y=np.pi, color='brown', label="$\pi$")
    main_ax.set_title("Buffon's Experiment Plot")
    main_ax.legend()
    return main_ax,

def update_fig(frame):

    estimated_pis.set_data(
        range(len(estimated_pis_list))[-50:], estimated_pis_list[-50:] # Show only last 50 points
                           )
    cummulative_pis.set_data(
        range(len(cummulative_pis_list)), cummulative_pis_list
    )

    # Adjust limits
    main_ax.relim()
    main_ax.autoscale_view()

    return estimated_pis, cummulative_pis


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
        with open(output_file, 'a') as file:
            file.write("#needles\t#crossed\tpi")

    logger.log(logging.INFO, parser.description)
    logger.log(logging.INFO, f"Saving values to {output_file}")


    def mqtt_on_message(client, userdata, message):
        logger.log(logging.INFO, f"Received message '{message.payload.decode()}'")
        match message.topic:
            case "buffon pi":
                msg = message.payload.decode()
                n, h, pi_value = msg.split("_")

                num_needles_list.append(int(n))
                num_crossed_list.append(max(int(h), 1)) # to avoid zero division
                estimated_pis_list.append(float(pi_value))
                cummulative_pis_list.append(
                    (2*sum(num_needles_list)) / sum(num_crossed_list)
                )

                with open(output_file, 'a') as file:
                    file.write("\n")
                    row = f"{n}\t{h}\t{pi_value}"
                    file.write(row)

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
