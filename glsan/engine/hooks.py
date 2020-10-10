import logging

import detectron2.utils.comm as comm
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.engine.hooks import (
    CallbackHook,
    IterationTimer,
    PeriodicWriter,
    PeriodicCheckpointer,
    LRScheduler,
    AutogradProfiler,
    EvalHook,
    PreciseBN
)
from detectron2.engine.hooks import EvalHook


class EvalHookRefine(EvalHook):

    def _do_eval(self):
        results = self._func()
        logger = logging.getLogger(__name__)

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            valid = dict()
            for k, v in flattened_results.items():
                try:
                    valid[k] = float(v)
                # currently only support skipping (List, Tensor, numpy.nda)
                # TODO: Maybe other types of Exceptions need to be taken into consideration
                except (ValueError, TypeError):
                    logger.info("Skip put {}: {} to tensorboard".format(k, type(v)))

            self.trainer.storage.put_scalars(**valid, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()
