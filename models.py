from keras.layers import Dense, Merge
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l2
from objectives import cca_loss


def create_model(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                    learning_rate, reg_par, outdim_size, use_all_singular_values):
    """
    builds the whole model
    the structure of each sub-network is defined in build_mlp_net,
    and it can easily get substituted with a more efficient and powerful network like CNN
    """
    view1_model = build_mlp_net(layer_sizes1, input_shape1, reg_par)
    view2_model = build_mlp_net(layer_sizes2, input_shape2, reg_par)

    model = Sequential()
    model.add(Merge([view1_model, view2_model], mode='concat'))

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(outdim_size, use_all_singular_values), optimizer=model_optimizer)

    return model


def build_mlp_net(layer_sizes, input_shape, reg_par):
    data_dim = input_shape[0]*input_shape[1]*input_shape[2]
    model = Sequential()
    for l_id, ls in enumerate(layer_sizes):
        if l_id == 0:
            input_dim = data_dim
        else:
            input_dim = []
        if l_id == len(layer_sizes)-1:
            activation = 'linear'
        else:
            activation = 'sigmoid'

        model.add(Dense(ls, input_dim=input_dim,
                                activation=activation,
                                W_regularizer=l2(reg_par),
                                b_regularizer=l2(reg_par)))
    return model
