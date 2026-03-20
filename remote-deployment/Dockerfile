# checkov:skip=CKV_DOCKER_2:Lambda manages container health checks
# checkov:skip=CKV_DOCKER_3:Lambda manages container user context
# nosemgrep: dockerfile-source-not-pinned  # Lambda base image, AWS manages updates
FROM public.ecr.aws/lambda/python:3.12-arm64

# Install dependencies (layer caching — this changes rarely)
COPY remote-deployment/requirements.txt /tmp/requirements.txt
RUN pip install --quiet --no-cache-dir -r /tmp/requirements.txt

# Copy handler
COPY remote-deployment/handler.py ${LAMBDA_TASK_ROOT}/

# Copy project source (no duplication — references repo root)
COPY playbook_resolver.py ${LAMBDA_TASK_ROOT}/
COPY runbook_functions.py ${LAMBDA_TASK_ROOT}/
COPY abstract_playbooks/ ${LAMBDA_TASK_ROOT}/abstract_playbooks/
COPY concrete_playbooks/ ${LAMBDA_TASK_ROOT}/concrete_playbooks/
COPY datasources/ ${LAMBDA_TASK_ROOT}/datasources/
COPY services/ ${LAMBDA_TASK_ROOT}/services/
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/
COPY playbooks/ ${LAMBDA_TASK_ROOT}/playbooks/

CMD ["handler.lambda_handler"]
