from datascience.ml.neural.reinforcement.game.offshore_regatta import OffshoreRegatta

game = OffshoreRegatta(source='grib_gfs_2018')
game.print()

# play
n_step = 700
for i in range(n_step):
    act, _ = game.bearing()
    game.action(act)
    if i % 100 == 0:
        print(i)
        game.print()

game.show_view()
game.plot(plot_weather=True, save=True)
print(game)
