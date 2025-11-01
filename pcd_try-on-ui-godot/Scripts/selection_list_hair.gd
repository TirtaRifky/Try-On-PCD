extends VBoxContainer

signal hair_selected(hair_type: int)

var button_list: Array[Button] = []
var connected_state := false

func _ready():
	# Button pada list_hair(VBoxContainer)
	for child in get_children():
		if child is Button:
			button_list.append(child)

	# Aktifkan on_pressed pada setiap tombol + disable klik sebelum koneksi
	for i in range(button_list.size()):
		var button = button_list[i]
		button.disabled = true
		button.focus_mode = Control.FOCUS_NONE
		button.connect("pressed", Callable(self, "_on_hair_button_pressed").bind(i))


# ============================= Event ======================================
func _on_hair_button_pressed(button_id: int) -> void:
	var hair_type = button_id + 1 
	print("💇 Hair Type %d clicked" % hair_type)
	emit_signal("hair_selected", hair_type)


# Pengubah status koneksi
func set_connected_state(state: bool) -> void:
	connected_state = state
	for button in button_list:
		button.disabled = not connected_state
		button.focus_mode = (Control.FOCUS_ALL if connected_state else Control.FOCUS_NONE)
