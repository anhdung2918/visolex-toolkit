import numpy as np

from visolex.utils import sort_data, evaluate, write_predictions, save_and_report_results
from .teacher import Teacher

class ViSoLexTrainer:
    def __init__(
        self, training_args,
        data_handler, tokenizer, normalizer, logger, evaluator,
        train_dataset, dev_dataset, test_dataset,
        unlabeled_dataset=None,
    ):
        if training_args.training_mode == "weakly_supervised":
            assert unlabeled_dataset is not None, \
                "Unlabeled dataset must be provided in Weakly Supervised Training"
        self.args = training_args
        self.dh = data_handler
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.logger = logger
        self.ev = evaluator
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled_dataset = unlabeled_dataset

        if self.args.training_mode != 'supervised':
            self.logger.info("Creating pseudo-dataset")
            self.pseudodataset = self.dh.create_pseudodataset(self.unlabeled_dataset)
            self.pseudodataset.downsample(self.args.sample_size)

        self.teacher_dev_res_list = []
        self.teacher_test_res_list = []
        self.teacher_train_res_list = []
        self.dev_res_list = []
        self.test_res_list = []
        self.train_res_list = []
        self.results = {}
        self.student_pred_list = []

    def train(self, write_pred=False):
        if self.args.training_mode == "supervised":
            self.supervised_training()
        elif self.args.training_mode == "semi_supervised":
            self.semi_supervised_training()
        elif self.args.training_mode == "weakly_supervised":
            self.weakly_supervised_training()

        # Store Final Results
        self.logger.info("Final Results")
        if self.args.training_mode == 'weakly_supervised':
            teacher_all_dev = [x['perf'] for x in self.teacher_dev_res_list]
            teacher_all_test = [x['perf'] for x in self.teacher_test_res_list]
            teacher_perf_str = [
                "{}:\t{:.2f}\t{:.2f}".format(
                    i, teacher_all_dev[i], teacher_all_test[i]) for i in np.arange(len(teacher_all_dev)
                )
            ]
            self.logger.info("TEACHER PERFORMANCES:\n{}".format("\n".join(teacher_perf_str)))

        all_dev = [x['perf'] for x in self.dev_res_list]
        all_test = [x['perf'] for x in self.test_res_list]
        perf_str = ["{}:\t{:.2f}\t{:.2f}".format(i, all_dev[i], all_test[i]) for i in np.arange(len(all_dev))]
        self.logger.info("STUDENT PERFORMANCES:\n{}".format("\n".join(perf_str)))

        # Get results in the best epoch (if multiple best epochs keep last one)
        best_dev_epoch = len(all_dev) - np.argmax(all_dev[::-1]) - 1
        best_test_epoch = len(all_test) - np.argmax(all_test[::-1]) - 1
        self.logger.info("BEST DEV {} = {:.3f} for epoch {}".format(
            self.args.metric, all_dev[best_dev_epoch], best_dev_epoch
        ))
        self.logger.info("FINAL TEST {} = {:.3f} for epoch {} (max={:.2f} for epoch {})".format(
            self.args.metric, all_test[best_dev_epoch], 
            best_dev_epoch, all_test[best_test_epoch], best_test_epoch
        ))

        if self.args.training_mode == 'weakly_supervised':
            self.results['teacher_train_iter'] = self.teacher_train_res_list
            self.results['teacher_dev_iter'] = self.teacher_dev_res_list
            self.results['teacher_test_iter'] = self.teacher_test_res_list

        self.results['student_train_iter'] = self.train_res_list
        self.results['student_dev_iter'] = self.dev_res_list
        self.results['student_test_iter'] = self.test_res_list

        self.results['student_dev'] = self.dev_res_list[best_dev_epoch]
        self.results['student_test'] = self.test_res_list[best_dev_epoch]
        if self.args.training_mode == 'weakly_supervised':
            self.results['teacher_dev'] = self.teacher_dev_res_list[best_dev_epoch]
            self.results['teacher_test'] = self.teacher_test_res_list[best_dev_epoch]

        if write_pred:
            write_predictions(
                self.args, self.logger, self.tokenizer, self.student_pred_list[best_dev_epoch], file_name="student_best_predictions"
            )
        
        # Save models and results
        self.normalizer.save("student_last")
        if self.args.training_mode == 'weakly_supervised':
            self.teacher.save("teacher_last")
        save_and_report_results(self.args, self.results, self.logger)

    def supervised_training(self):
        self.logger.info("\n\n\t*** Training Student on labeled data ***")

        newtraindataset = self.dh.create_pseudodataset(self.train_dataset)
        self.results['student_train'] = self.normalizer.train(
            train_dataset=newtraindataset, dev_dataset=self.dev_dataset, mode='train'
        )

        self.train_res_list.append(self.results['student_train'])
        if self.args.training_mode == 'supervised':
            self.normalizer.save('student_best')

        self.logger.info("\n\n\t*** Evaluating student on dev data ***")
        self.results['supervised_student_dev'] = evaluate(
            self.normalizer, self.dev_dataset, self.ev, 
            comment="student dev", remove_accents=self.args.remove_accents
        )
        self.dev_res_list.append(self.results['supervised_student_dev'])

        self.logger.info("\n\n\t*** Evaluating student on test data ***")
        self.results['supervised_student_test'], s_test_dict = evaluate(
            self.normalizer, self.test_dataset, self.ev, "test", 
            comment="student test", remove_accents=self.args.remove_accents
        )
        self.test_res_list.append(self.results['supervised_student_test'])
        self.student_pred_list.append(s_test_dict)

    def semi_supervised_training(self):
        self.supervised_training()

        for iter in range(self.args.num_iter):
            self.logger.info("\n\n\t *** Starting loop {}/{} ***".format(iter+1, self.args.num_iter))
            # Create pseudo-labeled dataset
            self.pseudodataset.downsample(self.args.sample_size)

            sorted_pseudodataset = sort_data(self.pseudodataset)
            student_pred_dict_unlabeled = self.normalizer.predict(dataset=sorted_pseudodataset)

            self.logger.info("Update unlabeled data with Student's predictions")
            self.pseudodataset.student_data['id'] = student_pred_dict_unlabeled['id']
            self.pseudodataset.student_data['input_ids'] = student_pred_dict_unlabeled['input_ids']
            self.pseudodataset.student_data['is_nsw'] = student_pred_dict_unlabeled['is_nsw']
            self.pseudodataset.student_data['align_index'] = student_pred_dict_unlabeled['align_index']
            self.pseudodataset.student_data['labels'] = student_pred_dict_unlabeled['preds']
            self.pseudodataset.student_data['proba'] = student_pred_dict_unlabeled['proba']
            self.pseudodataset.student_data['weights'] = [
                np.max(array, axis=-1) for array in student_pred_dict_unlabeled['proba']
            ]
            self.pseudodataset.drop(col='labels', value=-1, type='student')
            del student_pred_dict_unlabeled

            self.logger.info('Re-train student on pseudo-labeled instances provided by the teacher')
            train_res = self.normalizer.train(
                train_dataset=self.pseudodataset, dev_dataset=self.dev_dataset, mode='train_pseudo'
            )

            self.logger.info('Fine-tuning the student on clean labeled data')
            train_res = self.normalizer.train(train_dataset=self.newtraindataset, dev_dataset=self.dev_dataset, mode='finetune')
            self.train_res_list.append(train_res)

            self.logger.info("\n\n\t*** Evaluating student on dev data and update records ***")
            dev_res = evaluate(self.normalizer, self.dev_dataset, self.ev, comment="student dev iter{}".format(iter+1))
            self.logger.info("Student Dev performance on iter {}: {}".format(iter, dev_res['perf']))
            self.logger.info("\n\n\t*** Evaluating student on dev data and update records ***")
            test_res, s_test_dict = evaluate(self.normalizer, self.test_dataset, self.ev, "test", comment="student test iter{}".format(iter+1))
            self.logger.info("Student Test performance on iter {}: {}".format(iter, test_res['perf']))

            prev_max = max([x['perf'] for x in self.dev_res_list])
            if dev_res['perf'] > prev_max:
                self.logger.info("Improved dev performance from {:.2f} to {:.2f}".format(prev_max, dev_res['perf']))
                self.normalizer.save("student_best")
            self.dev_res_list.append(dev_res)
            self.test_res_list.append(test_res)
            self.student_pred_list.append(s_test_dict)

    
    def weakly_supervised_training(self):
        self.supervised_training()

        self.logger.info("Building teacher")
        self.teacher = Teacher(self.args, tokenizer=self.tokenizer, logger=self.logger)
        self.teacher.student = self.normalizer
        if self.args.student_name in ['visobert', 'phobert']:
            self.teacher.agg_model.xdim = self.normalizer.trainer.model.config.hidden_size
        else:
            self.teacher.agg_model.xdim = self.normalizer.trainer.model.config.d_model

        self.logger.info("Initializing teacher")
        self.results['teacher_train'] = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'perf': 0}
        self.results['teacher_dev'] = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'perf': 0}
        self.results['teacher_test'] = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'perf': 0}
        self.teacher_train_res_list.append(self.results['teacher_train'])
        self.teacher_dev_res_list.append(self.results['teacher_dev'])
        self.teacher_test_res_list.append(self.results['teacher_test'])

        for iter in range(self.args.num_iter):
            self.logger.info("\n\n\t *** Starting loop {}/{} ***".format(iter+1, self.args.num_iter))
            # Create pseudo-labeled dataset
            self.pseudodataset.downsample(self.args.sample_size)

            _ = self.teacher.train_ran(
                train_dataset=self.train_dataset, 
                dev_dataset=self.dev_dataset, 
                unlabeled_dataset=self.pseudodataset
            )

            # Apply Teacher on unlabeled data
            teacher_pred_dict_unlabeled = self.teacher.predict_ran(dataset=self.pseudodataset)

            self.logger.info("\n\n\t*** Evaluating teacher on dev data ***")
            teacher_dev_res, t_dev_dict = evaluate(self.teacher, self.dev_dataset, self.ev, "ran", comment="teacher dev iter{}".format(iter+1))
            teacher_dev_res_list.append(teacher_dev_res)
            self.logger.info("\n\n\t*** Evaluating teacher on test data ***")
            teacher_test_res, t_test_dict = evaluate(self.teacher, self.test_dataset, self.ev, "ran", comment="teacher test iter{}".format(iter+1))
            self.teacher_test_res_list.append(teacher_test_res)

            self.logger.info("Update unlabeled data with Teacher's predictions")
            self.pseudodataset.teacher_data['id'] = teacher_pred_dict_unlabeled['id']
            self.pseudodataset.teacher_data['input_ids'] = teacher_pred_dict_unlabeled['input_ids']
            self.pseudodataset.teacher_data['is_nsw'] = teacher_pred_dict_unlabeled['is_nsw']
            self.pseudodataset.teacher_data['align_index'] = teacher_pred_dict_unlabeled['align_index']
            self.pseudodataset.teacher_data['labels'] = teacher_pred_dict_unlabeled['preds']
            self.pseudodataset.teacher_data['proba'] = teacher_pred_dict_unlabeled['proba']
            self.pseudodataset.teacher_data['weights'] = [np.max(array, axis=-1) for array in teacher_pred_dict_unlabeled['proba']]
            self.pseudodataset.drop(col='labels', value=-1, type='teacher')
            del teacher_pred_dict_unlabeled


            self.logger.info('Re-train student on pseudo-labeled instances provided by the teacher')
            train_res = self.normalizer.train(train_dataset=self.pseudodataset, dev_dataset=self.dev_dataset, mode='train_pseudo')

            self.logger.info('Fine-tuning the student on clean labeled data')
            train_res = self.normalizer.train(train_dataset=self.newtraindataset, dev_dataset=self.dev_dataset, mode='finetune')
            self.train_res_list.append(train_res)

            self.logger.info("\n\n\t*** Evaluating student on dev data and update records ***")
            dev_res = evaluate(self.normalizer, self.dev_dataset, self.ev, comment="student dev iter{}".format(iter+1))
            self.logger.info("Student Dev performance on iter {}: {}".format(iter, dev_res['perf']))
            self.logger.info("\n\n\t*** Evaluating student on dev data and update records ***")
            test_res, s_test_dict = evaluate(self.normalizer, self.test_dataset, self.ev, "test", comment="student test iter{}".format(iter+1))
            self.logger.info("Student Test performance on iter {}: {}".format(iter, test_res['perf']))

            prev_max = max([x['perf'] for x in self.dev_res_list])
            if dev_res['perf'] > prev_max:
                self.logger.info("Improved dev performance from {:.2f} to {:.2f}".format(prev_max, dev_res['perf']))
                self.normalizer.save("student_best")
                self.teacher.save("teacher_best")
            self.dev_res_list.append(dev_res)
            self.test_res_list.append(test_res)
            self.student_pred_list.append(s_test_dict)