extends Control

# ========================================================
# Referensi Node UI
# ========================================================
@onready var hair_list_controller = $BoxContainer/MarginContainer/SplitContainer/ScrollContainer/ListHairContainer/MarginContainer/list_hair
@onready var button_connect = $BoxContainer/MarginContainer/SplitContainer/AspectRatioContainer/WebCam/MarginContainer/HBoxContainer/Button
@onready var status_label = $BoxContainer/MarginContainer/SplitContainer/AspectRatioContainer/WebCam/WebCamPanel/WebCamFeed/CameraStatusLabel
@onready var webcam_feed = $BoxContainer/MarginContainer/SplitContainer/AspectRatioContainer/WebCam/WebCamPanel/WebCamFeed
@onready var hair_display = $BoxContainer/MarginContainer/SplitContainer/AspectRatioContainer/WebCam/WebCamPanel/WebCamFeed/HairOverlay/HairDisplay
@onready var fps_label = $BoxContainer/MarginContainer/SplitContainer/AspectRatioContainer/WebCam/WebCamPanel/WebCamFeed/FpsInfo
@onready var hair_info = $BoxContainer/MarginContainer/SplitContainer/AspectRatioContainer/WebCam/WebCamPanel/WebCamFeed/HairInfo

# ========================================================
# Variabel Koneksi UDP Dummy
# ========================================================
var connected_state := false
var udp_client := PacketPeerUDP.new()
var server_ip := "127.0.0.1" # Localhost server (Dummy test)
var server_port := 9000     # Port server dummy

# ========================================================
# Style UI untuk tombol Connect/Disconnect
# ========================================================
var style_connect: StyleBoxFlat
var style_disconnect: StyleBoxFlat

# Batas waktu tunggu jawaban server (dalam frame)
# Jika 60 frame @60fps ≈ 1 detik
var connection_timeout_frames := 60


func _ready():
	print("=== HairController Ready ===")

	# Inisialisasi style box tombol Connect
	style_connect = StyleBoxFlat.new()
	style_connect.bg_color = Color(0.5, 0.9, 0.5, 0.9) # Hijau muda
	
	# Inisialisasi style box tombol Disconnect
	style_disconnect = StyleBoxFlat.new()
	style_disconnect.bg_color = Color(0.9, 0.5, 0.6, 0.9) # Merah muda
	
	button_connect.focus_mode = Control.FOCUS_NONE

	# Klik tombol Connect
	button_connect.connect("pressed", Callable(self, "_on_connect_pressed"))

	# Sinyal daftar rambut → Kirim data ke server UDP
	hair_list_controller.connect("hair_selected", Callable(self, "send_hair_selection"))

	# Status awal UI dalam keadaan tidak terkoneksi
	_set_button_state(false)
	hair_list_controller.set_connected_state(false)


# ========================================================
# Event klik tombol Connect/Disconnect
# ========================================================
func _on_connect_pressed():
	if connected_state:
		_disconnect_udp()
	else:
		await _connect_udp() # Proses koneksi async

func _process(_delta):
	if connected_state and udp_client.get_available_packet_count() > 0:
		var packet = udp_client.get_packet()
		# Only process large packets (likely image data)
		if packet.size() > 100:
			var webcam_node = $BoxContainer/MarginContainer/SplitContainer/AspectRatioContainer/WebCam
			webcam_node.process_camera_packet(packet)


# ========================================================
# Fungsi Dummy Koneksi UDP → kirim "ping"
# tunggu balasan: "pong" atau "ok"
# ========================================================
func _connect_udp():
	print("🔄 Attempting UDP connect...")
	status_label.text = "Status: Connecting..."

	# Mencoba membuka socket UDP ke server
	var err = udp_client.connect_to_host(server_ip, server_port)
	if err != OK:
		print("❌ UDP connect_to_host failed:", err)
		status_label.text = "Status: Connection failed"
		_set_button_state(false)
		return

	# Kirim pesan handshake sederhana
	var message = "ping"
	var send_result = udp_client.put_packet(message.to_utf8_buffer())
	if send_result != OK:
		print("❌ Failed to send UDP packet:", send_result)
		status_label.text = "Status: Send failed"
		udp_client.close()
		return

	print("📤 Sent to server:", message)

	# Tunggu balasan server dalam batas waktu
	var got_reply := false
	for i in range(connection_timeout_frames):
		await get_tree().process_frame
		if udp_client.get_available_packet_count() > 0:
			var packet = udp_client.get_packet()
			var reply = packet.get_string_from_utf8()

			# Jika balasan valid → koneksi berhasil
			if reply.begins_with("pong") or reply.begins_with("ok"):
				got_reply = true
				print("📥 Server replied:", reply)
				break

	# Jika server membalas → Set status connected
	if got_reply:
		print("✅ Server acknowledged")
		connected_state = true
		status_label.text = "Status: Connected"
		hair_list_controller.set_connected_state(true)
	else:
		print("⚠️ No reply from server.")
		udp_client.close()
		connected_state = false
		status_label.text = "Status: No response"
		hair_list_controller.set_connected_state(false)

	# Update style UI tombol
	_set_button_state(connected_state)


# ========================================================
# Fungsi Disconnect Dummy UDP
# Kirim pesan "disconnect" sebelum tutup socket
# ========================================================
func _disconnect_udp():
	print("🔌 Disconnected manually")

	if udp_client.is_socket_connected():
		var msg = "disconnect"
		var result = udp_client.put_packet(msg.to_utf8_buffer())
		if result == OK:
			print("📤 Sent disconnect message to server")
		else:
			print("⚠️ Failed to send disconnect message:", result)
	
	udp_client.close()
	connected_state = false
	status_label.text = "Status: Disconnected"
	hair_list_controller.set_connected_state(false)
	_set_button_state(false)


# ========================================================
# UI: Ubah style tombol sesuai status koneksi
# ========================================================
func _set_button_state(is_connected: bool):
	if is_connected:
		button_connect.text = "Disconnect"
		button_connect.add_theme_stylebox_override("normal", style_disconnect)
		button_connect.add_theme_stylebox_override("hover", style_disconnect)
		button_connect.add_theme_stylebox_override("pressed", style_disconnect)
	else:
		button_connect.text = "Connect"
		button_connect.add_theme_stylebox_override("normal", style_connect)
		button_connect.add_theme_stylebox_override("hover", style_connect)
		button_connect.add_theme_stylebox_override("pressed", style_connect)


# ========================================================
# Kirim pilihan rambut ke Server UDP (Dummy)
# Contoh format: "hair_type:2"
# ========================================================
func send_hair_selection(hair_type: int):
	if not connected_state:
		print("⚠️ Not connected. Cannot send hair data.")
		return

	var message = "hair_type:%d" % hair_type
	var send_result = udp_client.put_packet(message.to_utf8_buffer())

	if send_result == OK:
		print("📤 Sent hair selection:", message)
	else:
		print("❌ Failed to send hair selection:", send_result)
