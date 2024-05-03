import numpy as np
import torch


def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def contrast_loss(y, user_embedding, item_embedding):
    # Normalize the embeddings
    user_embedding = torch.nn.functional.normalize(user_embedding, dim=-1)
    item_embedding = torch.nn.functional.normalize(item_embedding, dim=-1)

    # Set temperature parameter
    tau = 0.07

    # Compute similarity scores
    scores = torch.matmul(user_embedding, item_embedding.t()) / tau

    # Subtract max for numerical stability
    scores -= scores.max()

    exp_scores = scores.exp()

    # Compute the loss
    loss = torch.log(exp_scores.sum(dim=1)) - scores[range(scores.shape[0]), y]
    loss = loss.mean()

    return loss


def fe_score(user_rep, item_rep, user_fea_col, item_fea_col, user_embedding_dim, item_embedding_dim):
    # print(user_rep.shape)
    # print(item_rep.shape)
    # print("col_score")
    score = []
    # user_embedding, item_embedding  = user_rep[0],item_rep[0]
    # user_rep = torch.reshape(user_embedding, (-1, user_fea_col, user_embedding_dim[0]))
    # item_rep = torch.reshape(item_embedding, (-1, item_fea_col, item_embedding_dim[0]))
    #
    # return (user_rep @ item_rep.permute(0, 2, 1)).max(2).values.sum(1)


    for i in range(len(user_embedding_dim)):
        # print(user_rep[i].shape)
        # print(item_rep[i].shape)
        user_temp = torch.reshape(user_rep[i], (-1, user_fea_col, user_embedding_dim[i]))
        item_temp = torch.reshape(item_rep[-1], (-1, item_fea_col, item_embedding_dim[i]))
        # print(user_temp.shape)
        # print(item_temp.shape)
        score.append((user_temp @ item_temp.permute(0, 2, 1)).max(2).values.sum(1))
    # all_score = 0.4 * score[0] + 0.2*score[1] + 0.4*score[2]
    score = torch.stack(score).transpose(1, 0)
    # # print(torch.sum(score,1))
    # all_score = all_score.unsqueeze(1)
    # # print(all_score.shape)
    # return all_score
    return torch.sum(score, 1)

