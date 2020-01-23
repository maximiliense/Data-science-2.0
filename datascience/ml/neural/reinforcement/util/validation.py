from engine.training_validation.prediction import play


def validation(model_z, output_size, game_class, game_params, nb_games):
    total_score = 0.
    for _ in range(nb_games):
        _, _, score = play(model_z, output_size, game_class, game_params)
        total_score += score
    return total_score / nb_games
