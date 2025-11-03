extends Control


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func _on_info_button_pressed():
	$"../Popup".visible = true # Replace with function body.


func _on_exit_button_pressed():
	$"../Popup".visible = false # Replace with function body.
