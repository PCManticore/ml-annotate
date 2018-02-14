from flask import jsonify, render_template, request
from flask_login import login_required

from annotator import app
from annotator.extensions import db
from annotator.models import (Dataset, DatasetLabelProbability, LabelEvent,
                              Problem, ProblemLabel)
from annotator.utils import assert_rights_to_problem


def _dataset_selector(problem, filter_trained_only=False):
    probabilities = db.select(
        [db.func.json_object_agg(
            DatasetLabelProbability.label_id,
            DatasetLabelProbability.probability
        )],
        from_obj=DatasetLabelProbability
    ).where(
        DatasetLabelProbability.data_id == Dataset.id
    ).correlate(Dataset.__table__).label('dataset_probabilities')

    label_created_at = (
        db.select(
            [db.func.to_char(
                LabelEvent.created_at,
                db.text("'YYYY-MM-DD HH24:MI:SS'")
            )],
            from_obj=LabelEvent
        ).where(LabelEvent.data_id == Dataset.id)
            .order_by(LabelEvent.created_at.desc())
            .limit(1)
            .correlate(Dataset.__table__)
            .label('label_created_at')
    )

    label_matches = db.select(
        [db.func.json_agg(db.func.json_build_array(
            LabelEvent.label_id,
            LabelEvent.label_matches
        ))],
        from_obj=LabelEvent
    ).where(
        LabelEvent.data_id == Dataset.id
    ).correlate(Dataset.__table__).label('label_matches')

    data = (
        db.session.query(
            Dataset.id,
            Dataset.free_text,
            Dataset.entity_id,
            Dataset.table_name,
            Dataset.meta,
            probabilities,
            Dataset.sort_value,
            label_matches,
            label_created_at
        )
            .filter(Dataset.problem_id == problem.id)
    )
    if filter_trained_only:
        data = data.join(LabelEvent.data).filter(
            Dataset.problem_id == problem.id
        ).group_by(Dataset.id, LabelEvent.id).distinct(Dataset.id)

    data = data.order_by(Dataset.id.asc()).all()
    problem_labels = db.session.query(
        ProblemLabel.id,
        ProblemLabel.label
    ).filter(
        ProblemLabel.problem == problem
    ).order_by(ProblemLabel.order_index).all()
    return data, problem_labels


@app.route('/<uuid:problem_id>/multi_class_batch_label', methods=['POST'])
def multi_class_batch_label(problem_id):
    problem = Problem.query.get(problem_id)
    assert_rights_to_problem(problem)

    data = request.get_json()
    ids = [str(x) for x in data['selectedIds']]
    if not ids:
        return jsonify(error='No ids selected')

    labels = (
        db.session.query(LabelEvent.id)
        .outerjoin(LabelEvent.data).filter(
            Dataset.id.in_(ids),
            Dataset.problem_id == problem.id
        )
        .all()
    )
    if labels:
        LabelEvent.query.filter(
            LabelEvent.id.in_([x[0] for x in labels])
        ).delete(synchronize_session='fetch')

    if data['label'] != 'undo':
        for label in problem.labels:
            for dataset_id in ids:
                db.session.add(LabelEvent(
                    label_id=label.id,
                    label_matches=str(label.id) == data['label'],
                    data_id=dataset_id
                ))
    db.session.commit()
    return jsonify(status='ok', labels_removed=len(labels))


@app.route('/<uuid:problem_id>/batch_label', methods=['POST'])
def batch_label(problem_id):
    problem = Problem.query.get(problem_id)
    assert_rights_to_problem(problem)

    data = request.get_json()
    ids = [str(x) for x in data['selectedIds']]
    if not ids:
        return jsonify(error='No ids selected')

    labels = (
        db.session.query(LabelEvent.id)
        .outerjoin(LabelEvent.data).filter(
            Dataset.id.in_(ids),
            Dataset.problem_id == problem.id,
            LabelEvent.label_id == data['label']
        )
        .all()
    )
    if labels:
        LabelEvent.query.filter(
            LabelEvent.id.in_([x[0] for x in labels])
        ).delete(synchronize_session='fetch')

    if data['value'] != 'undo':
        for dataset_id in ids:
            db.session.add(LabelEvent(
                label_id=data['label'],
                label_matches=data['value'],
                data_id=dataset_id
            ))

    db.session.commit()
    return jsonify(status='ok', labels_removed=len(labels))


@app.route('/<uuid:problem_id>/dataset')
@login_required
def dataset(problem_id):
    problem = Problem.query.get(problem_id)
    assert_rights_to_problem(problem)

    data, problem_labels = _dataset_selector(problem)

    return render_template(
        'dataset.html',
        data=data,
        problem=problem,
        problem_labels=problem_labels
    )


@app.route('/<uuid:problem_id>/dataset/trained')
@login_required
def trained_dataset(problem_id):
    problem = Problem.query.get(problem_id)
    assert_rights_to_problem(problem)

    data, problem_labels = _dataset_selector(
        problem,
        filter_trained_only=True,
    )

    return render_template(
        'dataset.html',
        data=data,
        problem=problem,
        problem_labels=problem_labels
    )
