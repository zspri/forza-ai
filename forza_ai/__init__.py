from imageai.Detection import ObjectDetection
from win32gui import GetForegroundWindow, GetWindowText
from . import datatypes, predict
import keyboard
import logging
import os
import pygame
import time

logging.basicConfig(level=logging.INFO)

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.abspath('retinanet.h5'))
detector.loadModel(detection_speed='faster')

custom_obj = detector.CustomObjects(car=True, bus=True, train=True, truck=True, traffic_light=True, stop_sign=True)

pygame.init()
font = pygame.font.SysFont('Consolas', 14)

scr_size = (960, 540)
center_point = (480, 410)
scr = pygame.display.set_mode(scr_size)
ui_modes = {
    0: 'lpr+overlay',  # lane prediction w/ screen grab
    1: 'lpr+seg',  # lane prediction w/ segmentation data
    2: 'recog'  # object recognition
}
ui_mode = 0


pr = predict.PredictionHandler()

print('Waiting for active window...')
fg = None
while not fg == 'Forza Horizon 4':
    fg = GetWindowText(GetForegroundWindow())

    # update pygame window so it doesn't freeze
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

    scr.fill((0, 0, 0))
    scr.blit(font.render('Waiting for active window...', True, (255, 255, 255)), (0, 0))
    pygame.display.flip()

print('Got window lock')

keyboard.press_and_release('escape')
time.sleep(1.5)

pr.thread.start()


def change_gui():
    global ui_mode
    ui_mode += 1
    if ui_mode > max(ui_modes.keys()):
        ui_mode = 0
    logging.info('changing gui mode - %s', ui_mode)


# keyboard.add_hotkey('ctrl+e', lambda: pr.data_queue.queue.clear())
keyboard.add_hotkey('ctrl+g', lambda: change_gui())
# keyboard.add_hotkey('ctrl+p', lambda: create_screenshot())


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            print(event)

    scr.fill((0, 0, 0))
    main_surf = pygame.Surface(scr_size)

    pr.do_canny = ui_mode != 2
    if ui_mode == 2:
        with pr.data_queue.mutex:
            pr.data_queue.queue.clear()
    prediction: datatypes.PredictionData = pr.data_queue.get()

    if ui_mode == 0:
        orig = prediction.original_as_pygame_image(scr_size)
        main_surf.blit(orig, orig.get_rect())
    elif ui_mode == 1:
        seg = prediction.segmentation_as_pygame_image(scr_size)
        main_surf.blit(seg, seg.get_rect())
    elif ui_mode == 2:
        img, recog = prediction.do_object_recognition(detector, custom_obj)
        main_surf.blit(img, img.get_rect())

        if len(recog) > 0:
            """
            primary_veh: list = [obj for obj in recog if obj.is_primary]
            primary_veh: datatypes.RecognizedObject = primary_veh[0] if len(primary_veh) > 0 else recog[0]
            """

            vehicle: datatypes.RecognizedObject
            for vehicle in recog:
                primary_rect: pygame.Rect = vehicle.rect  # primary.rect.inflate(-30, -30)
                primary_surf = pygame.Surface(primary_rect.size)
                primary_surf.fill((0, 0, 0))

                ratio = primary_rect.width / primary_rect.height
                side = 'side' if ratio > 1.6 else 'front'
                ratio_txt = font.render(f'{side}, {ratio}', True, (255, 255, 255))
                primary_surf.set_alpha(150)
                primary_surf.blit(ratio_txt, (0, 0))

                main_surf.blit(primary_surf, primary_rect)

                pygame.draw.polygon(main_surf, (0, 0, 255), [
                    center_point,
                    vehicle.rect.topleft,
                    (vehicle.rect.left, center_point[1])
                ], 1)

                pygame.draw.polygon(main_surf, (0, 0, 255), [
                    center_point,
                    vehicle.rect.topright,
                    (vehicle.rect.right, center_point[1])
                ], 1)

        pygame.draw.polygon(main_surf, (0, 255, 0), [
            (center_point[0] - 170, center_point[1]),
            (center_point[0] + 170, center_point[1]),
            (center_point[0], 250)
        ], 1)

        print(recog)

    if ui_mode in [0, 1]:
        vis = prediction.visualize_lines()
        vis = pygame.transform.scale(vis, scr_size)
        main_surf.blit(vis, vis.get_rect())

    ui_mode_txt = font.render('mode: ' + ui_modes[ui_mode], True, (255, 0, 0))
    main_surf.blit(ui_mode_txt, (0, 0))

    pygame.draw.rect(main_surf, (255, 0, 0), pygame.Rect(480 - 5, 410, 10, 10), 5)

    scr.blit(main_surf, (0, 0))
    pygame.display.flip()
