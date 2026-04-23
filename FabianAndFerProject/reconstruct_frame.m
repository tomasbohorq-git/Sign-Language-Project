% EMMAeye Timestamp-Based Frame Reconstructor
clc; clear; close all;


% CONFIGURATION
filename = 'emmaeye_log.jsonl';

% Set the target time you want to reconstruct. 
% (use time after midnight in your log file to find the frame you need)
TARGET_TIME_MS = 42636842; 

% Anatomical connection matrices
pose_bones = [12 14; 14 16; 13 15; 15 17; 12 13; 12 24; 13 25; 24 25];
hand_bones = [
    1 2; 2 3; 3 4; 4 5;       % Thumb
    1 6; 6 7; 7 8; 8 9;       % Index
    1 10; 10 11; 11 12; 12 13;% Middle
    1 14; 14 15; 15 16; 16 17;% Ring
    1 18; 18 19; 19 20; 20 21;% Pinky
    6 10; 10 14; 14 18        % Knuckles
];


% DATA LOADING & SEEKING (BY TIMESTAMP)
fid = fopen(filename, 'r');
if fid == -1
    error('Could not open %s. Ensure it is in the same directory.', filename);
end

target_data = [];
frames_checked = 0;

% Stream through the JSONL file until we hit the target timestamp
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line)
        continue; 
    end
    
    data = jsondecode(line);
    
    % Only analyze frames where people are actually detected
    if ~isempty(data.people)
        frames_checked = frames_checked + 1;
        
        % Check if the frame's timestamp meets or exceeds our target
        if data.ms_after_midnight >= TARGET_TIME_MS
            target_data = data;
            break; % Stop searching, we found our frame!
        end
    end
end
fclose(fid);

if isempty(target_data)
    error('Target time %d ms not found in the log file.', TARGET_TIME_MS);
end

fprintf('Successfully found frame! Target was %d ms, loaded frame is at %d ms.\n', ...
        TARGET_TIME_MS, target_data.ms_after_midnight);


% VISUALIZATION
figure('Name', sprintf('EMMAeye Reconstructor - Time: %d ms', target_data.ms_after_midnight), ...
       'Color', 'w', 'Position', [100, 100, 1000, 600]);
hold on; axis ij; grid on;

title(sprintf('EMMAeye Reconstructor | Simulator Time: %d ms after midnight', target_data.ms_after_midnight), 'FontSize', 14);
xlabel('X Pixels'); ylabel('Y Pixels');

for i = 1:length(target_data.people)
    person = target_data.people(i);
    
    % 1. DRAW POSE WIREFRAME
    if ~isempty(person.pose)
        % Safely extract the matrix whether it is a numeric array or a cell
        pose_lms = person.pose.landmarks;
        if iscell(pose_lms)
            pose_lms = cell2mat(pose_lms);
        end
        if size(pose_lms, 2) ~= 3
            pose_lms = pose_lms'; 
        end
        
        % Draw Bones
        for b = 1:size(pose_bones, 1)
            p1 = pose_bones(b, 1); p2 = pose_bones(b, 2);
            if p1 <= size(pose_lms, 1) && p2 <= size(pose_lms, 1)
                plot([pose_lms(p1,1), pose_lms(p2,1)], [pose_lms(p1,2), pose_lms(p2,2)], 'b-', 'LineWidth', 2);
            end
        end
        % Draw Joints
        scatter(pose_lms(:,1), pose_lms(:,2), 40, 'b', 'filled');
        text(person.pose.center(1) - 30, person.pose.center(2) - 40, ...
            sprintf('Person %d', person.id), 'Color', 'b', 'FontSize', 12, 'FontWeight', 'bold');
    end
    
    % 2. DRAW FACE
    if ~isempty(person.face)
        bbox = person.face.bbox; 
        rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 2);
    end
    
    % 3. DRAW HANDS & GESTURES
    sides = {'BODY_LEFT', 'BODY_RIGHT'};
    colors = {'m', 'c'}; % Magenta for Left, Cyan for Right
    
    for s = 1:2
        side = sides{s};
        if isfield(person.hands, side) && ~isempty(person.hands.(side))
            hand_data = person.hands.(side);
            
            % Safely extract the hand matrix
            lms = hand_data.lm;
            if iscell(lms)
                lms = cell2mat(lms);
            end
            if size(lms, 2) ~= 3
                lms = lms'; 
            end
            
            % Draw Hand Bones
            for b = 1:size(hand_bones, 1)
                p1 = hand_bones(b, 1); p2 = hand_bones(b, 2);
                plot([lms(p1,1), lms(p2,1)], [lms(p1,2), lms(p2,2)], 'Color', colors{s}, 'LineWidth', 1.5);
            end
            
            % Draw Knuckles
            scatter(lms(:,1), lms(:,2), 20, colors{s}, 'filled');
            
            % Draw Bounding Box and Gesture Label
            h_bbox = hand_data.bbox;
            rectangle('Position', h_bbox, 'EdgeColor', colors{s}, 'LineStyle', '--');
            text(h_bbox(1), h_bbox(2) - 10, ...
                sprintf('G%d', hand_data.stable_gesture), 'Color', colors{s}, 'FontWeight', 'bold', 'FontSize', 11);
        end
    end
end

xlim([0 1280]);
ylim([0 720]);
hold off;