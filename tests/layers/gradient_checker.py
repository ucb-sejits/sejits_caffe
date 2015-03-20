import numpy as np


class GradientChecker(object):
    def __init__(self, stepsize, threshold, seed=1701, kink=0, kink_range=-1):
        self.stepsize = stepsize
        self.threshold = threshold
        self.seed = seed
        self.kink = kink
        self.kink_range = kink_range

    def get_obj_and_gradient(self, layer, top, top_diff, top_id, top_data_id):
        loss = 0
        if top_id < 0:
            for top_data, diff in zip(top, top_diff):
                loss += np.sum(np.square(top_data))
                diff[:] = top_data
            loss /= 2
        else:
            for diff in top_diff:
                diff.fill(0)
            loss_weight = 2
            loss = top[top_id].ravel()[top_data_id] * loss_weight
            top[top_id].ravel()[top_data_id] = loss_weight
        return loss

    def check_gradient_single(self, layer, bottom, bottom_diff, top, top_diff,
                              check_bottom, top_id, top_data_id,
                              element_wise=False):
        if element_wise:
            assert len(layer.blobs) == 0
            assert 0 <= top_id
            assert 0 <= top_data_id
            top_count = top[top_id].size()
            for bottom_data in bottom:
                assert top_count == bottom_data.size()

        blobs_to_check = layer.blobs[:]

        if check_bottom < 0:
            for bottom_data in bottom:
                blobs_to_check.append(bottom_data)
        else:
            assert check_bottom < bottom.shape[0]
            blobs_to_check.append(bottom[check_bottom])

        layer.forward(bottom, top)
        self.get_obj_and_gradient(layer, top, top_diff, top_id, top_data_id)
        layer.backward(bottom_diff, top_diff)

        computed_gradients = []
        for blob, computed in zip(blobs_to_check, computed_gradients):
            for feat_id in range(blob.shape[0]):
                estimated_gradient = 0
                positive_objective = 0
                negative_objective = 0
                if not element_wise or feat_id == top_data_id:
                    blob[feat_id] += self.step_size
                    layer.forward(bottom, top)
                    positive_objective = self.get_obj_and_gradient(layer, top,
                                                                   top_diff,
                                                                   top_id,
                                                                   top_data_id)
                    blob[feat_id] -= self.step_size * 2
                    layer.forward(bottom, top)
                    negative_objective = self.get_obj_and_gradient(layer, top,
                                                                   top_diff,
                                                                   top_id,
                                                                   top_data_id)

                    blob += self.stepsize
                    estimated_gradient = positive_objective - \
                        negative_objective / self.stepsize / 2

                computed_gradient = computed_gradients[feat_id]
                feature = blob[feat_id]
                if (self.kink - self.kink_range > abs(feature) or abs(feature)
                        > self.kink + self.kink_range):
                    scale = max(max(abs(computed_gradient),
                                    abs(estimated_gradient)), 1)
                    assert abs(computed_gradient - estimated_gradient) < \
                        self.threshold * scale

    def check_gradient_exhaustive(self, layer, bottom, bottom_diff, top,
                                  top_diff, check_bottom=-1):
        layer.setup(bottom, top)
        assert top.shape[0] > 0
        for i in range(top.shape[0]):
            for j in range(top.size):
                self.check_gradient_single(layer, bottom, bottom_diff, top,
                                           top_diff, check_bottom, i,
                                           j)
