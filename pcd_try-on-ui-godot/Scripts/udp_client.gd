extends Node
class_name UdpClient

signal connected
signal disconnected
signal message_received(message: String)
signal connection_failed

var udp := PacketPeerUDP.new()
var connected_state := false
var server_ip := "127.0.0.1"
var server_port := 4242  # sesuai dengan server Python
var _listening := false


func connect_to_server() -> void:
	print("🔄 Connecting to Python UDP stream server on %s:%d..." % [server_ip, server_port])
	var err = udp.connect_to_host(server_ip, server_port)
	if err != OK:
		print("❌ Failed to connect:", err)
		emit_signal("connection_failed")
		return

	connected_state = true
	emit_signal("connected")

	# Kirim pesan awal (handshake)
	_send_text("ping")
	print("📤 Sent handshake: ping")

	# Mulai thread listener
	if not _listening:
		_listening = true
		_listen_udp()


func _listen_udp() -> void:
	print("👂 Listening for UDP messages...")
	while connected_state:
		await get_tree().process_frame
		if udp.get_available_packet_count() > 0:
			var data = udp.get_packet().get_string_from_utf8()
			emit_signal("message_received", data)


func send_message(message: String) -> void:
	if not connected_state:
		print("⚠️ Not connected. Cannot send message.")
		return
	_send_text(message)


func _send_text(text: String) -> void:
	var result = udp.put_packet(text.to_utf8_buffer())
	if result == OK:
		print("📤 Sent:", text)


func disconnect_from_server() -> void:
	if connected_state:
		print("🔌 Disconnecting from server...")
		_send_text("disconnect")
		udp.close()
	connected_state = false
	_listening = false
	emit_signal("disconnected")
