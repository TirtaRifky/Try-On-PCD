extends VBoxContainer

@onready var webcam_feed = $WebCamPanel/WebCamFeed
@onready var fps_label = $WebCamPanel/WebCamFeed/FpsInfo
@onready var status_label = $WebCamPanel/WebCamFeed/CameraStatusLabel

var image = Image.new()
var texture = ImageTexture.new()
var last_frame_time = Time.get_ticks_msec()
var frame_count = 0
var fps = 0

func _ready():
	webcam_feed.texture = texture
	
func process_camera_packet(packet: PackedByteArray):
	# Create image from received JPEG data
	var error = image.load_jpg_from_buffer(packet)
	if error == OK:
		# Update texture with new frame
		texture = ImageTexture.create_from_image(image)
		webcam_feed.texture = texture
		
		# Update FPS
		frame_count += 1
		var current_time = Time.get_ticks_msec()
		var elapsed = current_time - last_frame_time
		
		if elapsed > 1000:  # Update FPS every second
			fps = frame_count * 1000.0 / elapsed
			fps_label.text = "FPS: %.1f" % fps
			frame_count = 0
			last_frame_time = current_time
