from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from common.config import Settings
from dataset.create_data import create_sc_dataset
from experiment import Experiment, Record, RecordContainer
from figure_generater.plot_config import PlotConfig
from loss import objective_val
from models import initialize_model

__all__ = ['MSELossExperiment', 'SuccessRateExperiment']


class MSELossExperiment(Experiment):
    desc: str = 'MSE Loss Experiment'

    def plot(self, plot_config: PlotConfig, total_results1: RecordContainer, total_results2: RecordContainer,save_path:str='./save.png'):
        plot_color = plot_config.plot_color

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.subplots_adjust(hspace=0.5)  #
        for index, record in enumerate(total_results1.results):
            linestyle = '--' if 'Joint' in record.desc else '-'
            ax1.plot(record.metrics, color=plot_color[index], linestyle=linestyle,
                     label=record.params.get('latex_name', record.desc))
            ax1.set_yscale('log')
            ax1.set_yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 10 ** -5, 10 ** -6, 10 ** -7],
                           [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$',
                            r'$10^{-6}$', r'$10^{-7}$'])

        ax1.set_ylabel('Relative Error', fontdict={'fontsize': 12})
        ax1.set_xlabel('Iteration', fontdict={'fontsize': 12})

        for index, record in enumerate(total_results2.results):
            linestyle = '--' if 'Joint' in record.desc else '-'
            ax2.plot(record.metrics, color=plot_color[index], linestyle=linestyle,
                     label=record.params.get('latex_name', record.desc))
            ax2.set_yscale('log')
            ax2.set_yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 10 ** -5, 10 ** -6, 10 ** -7],
                           [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$',
                            r'$10^{-6}$', r'$10^{-7}$'])
        ax2.set_ylabel('Relative Error', fontdict={'fontsize': 12})
        ax2.set_xlabel('Iteration', fontdict={'fontsize': 12})
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.52, 0.97), ncol=len(total_results1.results),
                   fontsize=12)

        # 显示图形
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(save_path)

    def run(self, logger, opts: Settings, model_prox_dict) -> RecordContainer:
        """Run the experiment and record results."""
        logger.info(f'Starting {self.desc}, Sparsity: {opts.sparsity}')
        container = RecordContainer(desc=self.desc, gloabl_params=opts.model_dump())

        (x_test, d_test), A, b = create_sc_dataset(opts=opts)

        for model_name, prox_func_list in model_prox_dict.items():
            for prox_func in prox_func_list:

                model = initialize_model(model_name, prox_func, A, opts)

                logger.info('Running ' + model.desc())
                model(d_test, K=opts.K)

                record = Record(desc=model.desc(), model_name=model.__module__,
                                params={'latex_name': self._get_prox_func_latex_name(prox_func)})

                for k in range(0, opts.K):
                    testing_loss = np.mean(objective_val(
                        model.iter_history[k], d_test, x_test, objective=opts.objective
                    ).item())  # Compute the average of the losses
                    record.metrics.append(testing_loss)
                    logger.info("Test loss", iteration=k, loss=testing_loss)

                container.results.append(record)

        return container

    @classmethod
    def _get_prox_func_latex_name(cls, prox_func):
        if isinstance(prox_func, tuple):
            return f'{prox_func[0].latex_name}+{prox_func[1].latex_name}'
        return f'{prox_func.latex_name}'


class SuccessRateExperiment(Experiment):



    desc: str = 'Success Rate Experiment'

    def run(self, logger, opts: Settings, model_prox_dict, sparsity_scope: Iterable, model_name: str = 'GROUPPGAC'):
        logger.info(f'Starting {self.desc}')
        container = RecordContainer(desc=self.desc, gloabl_params=opts.model_dump())
        model_name, prox_func_list = list(model_prox_dict.items())[0]

        for per_sparsity in sparsity_scope:

            opts.sparsity = per_sparsity
            logger.info('Using sparsity: {} '.format(per_sparsity) + '-- Sparse Level --> ' + str(
                per_sparsity / (opts.n / opts.gLen)))

            # Create data
            (x_test, d_test), A, b = create_sc_dataset(opts=opts)

            model = initialize_model(model_name, prox_func_list[0], A, opts)

            logger.info('Running ' + model.desc())

            model(d_test, K=opts.K)

            success_times = objective_val(model.iter_history[-1], d_test, x_test, objective='SUCCESS_TIMES')
            logger.info(f'Avg Success Times (success rate) {success_times}')

            record = Record(desc=model.desc(), model_name=model.__module__)
            record.metrics.append(success_times)
            container.results.append(record)
            if success_times == 0:
                break
        return container

    def plot(self,save_path:str='./save.png'):
        pass