from __future__ import annotations

import logging
from typing import Any, TypeVar

import flask
from flask import Flask
from gevent.pywsgi import WSGIServer

from musc.service.a import Service as ServiceLowLevel
from musc.service.errors import ServiceErrorByDuplicatedId, ServiceErrorFromModelPrediction
from musc.service.concepts.a import UpdateStrategy
from musc.service.concepts.high_level.a import apply_model_adaptors
from musc.service.impl.dd_process.stats import Stats


T = TypeVar('T')


class Service:

    def __init__(
        self,
        model: Any,
        update_strategy: UpdateStrategy,
        *,
        wait_y_timeout: float | None = None,
        daemon: bool = True,
        debug: bool = False,
    ) -> None:
        self._inner = ServiceLowLevel(
            apply_model_adaptors(model),
            update_strategy.clone_without_state(),
            wait_y_timeout=wait_y_timeout,
            daemon=daemon,
            debug=debug,
        )

    def recv_x(self, x: Any, id_: Any) -> Any:
        return self._inner.recv_x(x, str(id_))

    def recv_y(self, y: Any, id_: Any) -> None:
        return self._inner.recv_y(y, str(id_))

    def stats(self) -> Stats:
        return self._inner.stats()

    def stop_listening(self) -> None:
        return self._inner.stop_listening()

    def join(self) -> None:
        return self._inner.join()


def run_service_http(
    model: Any,
    update_strategy: UpdateStrategy,
    *,
    host: str = '127.0.0.1',
    port: int = 80,
    wait_y_timeout: float | None = None,
    debug: bool = False,
) -> None:

    model_ = apply_model_adaptors(model)
    service = Service(model_, update_strategy, wait_y_timeout=wait_y_timeout, debug=debug)

    flask_app = Flask(__name__)

    @flask_app.route('/', methods=['POST'])
    def recv_data() -> Any:

        request_json = flask.request.get_json(silent=True)
        if request_json is None:
            flask.abort(415)

        try:
            id_ = request_json['id']
        except KeyError:
            flask.abort(422)

        try:
            x = request_json['x']
        except KeyError:
            try:
                y = request_json['y']
            except KeyError:
                flask.abort(422)
            else:
                x, y, kind = None, model_.preprocess_y(y), 'y'
        else:
            try:
                y = request_json['y']
            except KeyError:
                x, y, kind = model_.preprocess_x(x), None, 'x'
            else:
                flask.abort(422)

        if kind == 'x':
            try:
                y_pred = service.recv_x(x, id_)
            except ServiceErrorByDuplicatedId:
                return {
                    'error':
                        f'Duplicated sample ID: {id_}. '
                        f'The service received an x-value with the same ID before.'
                }
            except ServiceErrorFromModelPrediction as e:
                basic_description = 'The underlying model prediction function raised an exception'
                logging.error(f'{basic_description}: {repr(e)}.')
                return {'error': f'{basic_description}.'}
            else:
                return {'y_pred': model_.y_pred_to_json_like(y_pred)}
        elif kind == 'y':
            try:
                service.recv_y(y, id_)
            except ServiceErrorByDuplicatedId:
                return {
                    'error':
                        f'Duplicated sample ID: {id_}. '
                        f'The service received a y-value with the same ID before.'
                }
            else:
                return {}
    del recv_data

    WSGIServer((host, port), flask_app).serve_forever()


__all__ = [
    'Service',
    'run_service_http',
]
