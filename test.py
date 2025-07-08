from pyboy import PyBoy


pyboy = PyBoy('Super Mario Bros. Deluxe (USA, Europe) (Rev 1).gbc')

while pyboy.tick():
    # This loop will run until the game is closed
    pass
pyboy.stop()  # Stop the emulator when done