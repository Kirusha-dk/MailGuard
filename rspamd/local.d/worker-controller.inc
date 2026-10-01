# BENCHMARK ONLY.
# The Docker port is bound to 127.0.0.1 on the host in docker-compose.yml.
# This lets the local benchmark runner call /learnspam and /learnham without
# placing a reusable password in the repository. Do not copy this setting to
# a production mail server.
bind_socket = "*:11334";
secure_ip = "0.0.0.0/0";
