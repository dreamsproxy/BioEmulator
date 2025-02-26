import pygame

# Constants
grid_size = 51  # Increase number of cells
default_cell_size = 20  # Reduce size for testing performance
view_width, view_height = 800, 600  # Initial window size
min_cell_size, max_cell_size = 5, 50  # Zoom limits

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Cell types
EMPTY = 0
DIAGONAL_TL_BR = 1
DIAGONAL_TR_BL = 2
UNFILLED_SQUARE = 3
FILLED_SQUARE = 4

UNFILLED_MERGED = 5

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((view_width, view_height), pygame.RESIZABLE)
pygame.display.set_caption("Graphing Paper")

cell_size = default_cell_size
grid = [[EMPTY for _ in range(grid_size)] for _ in range(grid_size)]
grid[((grid_size-1)//2)+1][((grid_size-1)//2)+1] = FILLED_SQUARE

offset_x, offset_y = 0, 0  # Offset for panning
dragging = False
last_mouse_pos = (0, 0)
fullscreen = False

def draw_grid():
    screen.fill(WHITE)
    visible_x_start = max(0, (-offset_x) // cell_size)
    visible_x_end = min(grid_size, (view_width - offset_x) // cell_size + 1)
    visible_y_start = max(0, (-offset_y) // cell_size)
    visible_y_end = min(grid_size, (view_height - offset_y) // cell_size + 1)
    
    merged_cells = set()
    
    for y in range(visible_y_start, visible_y_end - 1):
        for x in range(visible_x_start, visible_x_end - 1):
            # If pos between x -> x + 1 and y -> y +1 is UNFILLED_SQUARE
            if (grid[y][x] == UNFILLED_SQUARE and grid[y][x+1] == UNFILLED_SQUARE and
                grid[y+1][x] == UNFILLED_SQUARE and grid[y+1][x+1] == UNFILLED_SQUARE):
                merged_cells.add((x, y))

    
    for y in range(visible_y_start, visible_y_end):
        for x in range(visible_x_start, visible_x_end):
            if (x, y) in merged_cells:
                rect = pygame.Rect(
                    x * cell_size + offset_x,
                    y * cell_size + offset_y,
                    2 * cell_size,
                    2 * cell_size)
                pygame.draw.rect(screen, BLACK, rect, 1)

            # If where we clicked does not exist in merged cell, cycle as expected
            elif (x-1, y) not in merged_cells\
                and (x, y-1) not in merged_cells\
                and (x-1, y-1) not in merged_cells:
                rect = pygame.Rect(x * cell_size + offset_x, y * cell_size + offset_y, cell_size, cell_size)
                pygame.draw.rect(screen, GRAY, rect, 1)
                
                if grid[y][x] == DIAGONAL_TL_BR:
                    pygame.draw.line(screen, BLACK, rect.topleft, rect.bottomright, 2)
                elif grid[y][x] == DIAGONAL_TR_BL:
                    pygame.draw.line(screen, BLACK, rect.topright, rect.bottomleft, 2)
                elif grid[y][x] == UNFILLED_SQUARE:
                    pygame.draw.rect(screen, BLACK, rect, 1)
                elif grid[y][x] == FILLED_SQUARE:
                    pygame.draw.rect(screen, BLACK, rect)

def get_cell(pos):
    x, y = pos
    grid_x = (x - offset_x) // cell_size
    grid_y = (y - offset_y) // cell_size
    return grid_x, grid_y

def handle_click(pos, button, modifiers):
    x, y = get_cell(pos)
    if 0 <= x < grid_size and 0 <= y < grid_size:
        if button == 1 and not (modifiers & pygame.KMOD_ALT):  # Left click toggles filled/unfilled square
            if modifiers & pygame.KMOD_SHIFT:
                grid[y][x] = EMPTY  # Shift + Left Click clears the cell
            else:
                if grid[y][x] == FILLED_SQUARE:
                    grid[y][x] = UNFILLED_SQUARE
                else:
                    grid[y][x] = FILLED_SQUARE
        elif button == 3 and not (modifiers & pygame.KMOD_ALT):  # Right click cycles diagonal orientations
            if grid[y][x] in [EMPTY, UNFILLED_SQUARE, FILLED_SQUARE]:
                grid[y][x] = DIAGONAL_TL_BR
            elif grid[y][x] == DIAGONAL_TL_BR:
                grid[y][x] = DIAGONAL_TR_BL
            else:
                grid[y][x] = DIAGONAL_TL_BR

def zoom_grid(mouse_pos, zoom_in):
    global cell_size, offset_x, offset_y
    old_cell_size = cell_size
    if zoom_in:
        cell_size = min(cell_size + 2, max_cell_size)
    else:
        cell_size = max(cell_size - 2, min_cell_size)
    
    # Adjust offset to keep zoom centered around mouse position
    mx, my = mouse_pos
    scale_factor = cell_size / old_cell_size
    offset_x = int(mx - (mx - offset_x) * scale_factor)
    offset_y = int(my - (my - offset_y) * scale_factor)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode((view_width, view_height), pygame.RESIZABLE)
            elif event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.VIDEORESIZE:
            view_width, view_height = event.w, event.h
            screen = pygame.display.set_mode((view_width, view_height), pygame.RESIZABLE)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and pygame.key.get_mods() & pygame.KMOD_ALT:
                dragging = True
                last_mouse_pos = event.pos
            elif event.button == 4:  # Scroll up to zoom in
                zoom_grid(event.pos, True)
            elif event.button == 5:  # Scroll down to zoom out
                zoom_grid(event.pos, False)
            else:
                handle_click(event.pos, event.button, pygame.key.get_mods())
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                offset_x += dx
                offset_y += dy
                last_mouse_pos = event.pos
    
    draw_grid()
    pygame.display.flip()

pygame.quit()
