from engine.util.print_colors import color
from engine.training_validation.validation import validation
from engine.models.util import save_model
from engine.util.log_email import send_email
from engine.util.log_file import save_file


def training_logs(norm_exp, norm_opt, running_loss, running_reward, loss_logs, rewards_logs, epoch,
                  model_z, model_path):
    if norm_exp != 0:
        running_reward /= float(norm_exp)
    if norm_opt != 0:
        running_loss /= float(norm_opt)
    print('[%d] loss: %.5f, rewards: %f' % (epoch + 1, running_loss, running_reward))

    # logging the loss and the rewards
    loss_logs.append(running_loss)
    rewards_logs.append(running_reward)

    save_model(model_z, model_path)
    print(color.RED + 'Saved  model: ' + str(model_path) + color.END)


def validation_logs(epoch, validation_modulo, val_logs, send_results, save_results, model_z, output_size, game_class,
                    game_params, nb_validation_game, xp_name, validation_txt_path, root_dir,
                    save_validation_model=False, final=False):
    # constructing the validation number (first validation is 1, second is 2, etc.)
    if not final:
        validation_id = str(int((epoch + 1) / validation_modulo))

    # computing the validation score
    validation_score = validation(model_z, output_size, game_class, game_params, nb_validation_game)

    # append the validation to the logs, to draw a chart
    val_logs.append(validation_score)

    # validation message
    print(color.BLUE)
    if not final:
        val_str = '[Val: ' + validation_id + '] Validation score (over '
    else:
        val_str = 'Validation score (over '
    val_str += str(nb_validation_game) + ' games): ' + str(validation_score)
    print(val_str)
    print(color.END)

    # send and save validation results
    if send_results >= 1:
        if final:
            send_email('Final results for XP ' + xp_name, val_str)
        elif send_results == 2:
            send_email('Results for XP ' + xp_name + ' (epoch: ' + str(epoch + 1) + ')', val_str)
    if save_results:
        if final:
            save_file(validation_txt_path, 'Final results for XP ' + xp_name, val_str)
        else:
            save_file(validation_txt_path, 'Results for XP ' + xp_name + ' (epoch: ' + str(epoch + 1) + ')', val_str)

    # if not final save the tmp model
    if not final and save_validation_model:
        # save model used for validation
        model_validation_path = root_dir + '/' + xp_name + '_' + validation_id + '_model.torch'
        save_model(model_z, model_validation_path)
        print(color.RED + 'Saved validated model: ' + str(model_validation_path) + color.END)


def epoch_title(log_modulo, epoch, epsilon, scheduler):
    title_str = '\n' + color.GREEN + '-' * 5 + ' Epoch'
    if log_modulo == 1:
        title_str += ' ' + str(epoch + 1)
    else:
        title_str += 's ' + str(epoch + 1) + '-' + str(epoch + log_modulo)
    title_str += ' (lr: ' + str(scheduler.get_lr())
    title_str += ', eps: ' + str(epsilon) + ') ' + '-' * 5 + color.END
    return title_str
