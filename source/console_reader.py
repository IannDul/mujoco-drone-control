from threading import Thread

from source.models import Position
from source.wp_queue import WayPointsQueue


def start_console_reader_thread(wp_queue: WayPointsQueue):
    def target():
        print("Commands:\n "
              "> add x y z yaw -> insert new waypoint to queue\n "
              "> status -> returns current active waypoint\n"
              )

        while True:
            try:
                command = input('> ').strip()
            except KeyboardInterrupt:
                print('Finished!')
                break
            except EOFError:
                print('Error!')
                break

            if not command:
                continue

            parts = command.split(' ')
            command_name = parts[0].lower()

            match command_name:
                case 'add':
                    if len(parts) != 5:
                        print("Wrong number of params")
                        continue
                    try:
                        position = Position((float(parts[1]), float(parts[2]), float(parts[3])), float(parts[4]))
                        wp_queue.push(position)
                    except ValueError:
                        print("Wrong format of numbers")
                        continue
                case 'status':
                    cur_wp = wp_queue.get_current()
                    x, y, z = cur_wp.xyz
                    print(f'Current waypoint: \n'
                          f'x={x}, y={y}, z={z}, yaw={cur_wp.yaw}')
                case _:
                    print("Command not found")
                    continue

    thread = Thread(target=target, daemon=True)
    thread.start()
