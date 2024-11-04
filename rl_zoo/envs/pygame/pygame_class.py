import pygame as pg
import time
import random
pg.init()


class Moving_Auruco:
    """
        A class for miving the auruco code
    """
    def __init__(self):
        # Runtime
        self.auruco_number = 7

        self.velocity_x = 0
        self.velocity_y = 0
        self.location_x = 0
        self.location_y = 0

        self.time_interval = 0.8

        self.change_direction_x = False
        self.change_direction_y = False

        self.screen_height = 720
        self.screen_width = 1080
        self.auruco_x = 100
        self.auruco_y = 100
        self.margin_x = self.auruco_x / 2
        self.margin_y = self.auruco_y / 2
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.img = pg.image.load("/Users/tonyq/Downloads/4x4_354_7.png")
        self.img = pg.transform.scale(self.img, (self.auruco_x, self.auruco_y))

    def compute_location(self):
        # Deal with change direction first
        if self.change_direction_x:
            self.velocity_x *= -1
            self.change_direction_x = False
        if self.change_direction_y:
            self.velocity_y *= -1
            self.change_direction_y = False

        # Compute the next location of the auruco code.
        next_x = self.location_x + self.time_interval * self.velocity_x
        next_y = self.location_y + self.time_interval * self.velocity_y

        # Decide whether to hit the bound and change direction
        if next_x > (self.screen_width - self.auruco_x):
            next_x = self.screen_width - self.auruco_x
            self.change_direction_x = True

        if next_x < 0:
            next_x = 0
            self.change_direction_x = True

        if next_y > (self.screen_height - self.auruco_y):
            next_y = (self.screen_height - self.auruco_y)
            self.change_direction_y = True

        if next_y < 0:
            next_y = 0
            self.change_direction_y = True

        self.location_x = next_x
        self.location_y = next_y
        return next_x, next_y

    def run(self):
        previous_time = time.time()

        self.velocity_x = random.randint(10, 50)
        self.velocity_y = random.randint(10, 50)

        # Main loop
        while True:
            current_time = time.time()
            if (current_time - previous_time) < self.time_interval:
                continue
            code_x, code_y = self.compute_location()
            # Display
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.img, (code_x, code_y))
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    break
            # Update
            pg.display.update()

            previous_time = current_time


if __name__ == '__main__':
    auruco = Moving_Auruco()
    auruco.run()
