% Step 1: Load an image
img = imread('logo.jpg');
X = double(img); % Convert to double for processing

% Define different ranks to test
ranks = [100, 70, 50, 30, 10]; % Different ranks for comparison
memory_usages = zeros(size(ranks));
losses = zeros(size(ranks));
original_memory = numel(X) * 8;

% Create a figure and set its size
figure;
set(gcf, 'Units', 'pixels', 'Position', [100, 100, 800, 1200]);  % Adjust width and height as needed

% Initialize a container for subplot handles for adjusting margins later
hSub = zeros(length(ranks) + 1, 1);

% Display the original image first
hSub(1) = subplot(length(ranks)+1, 1, 1);
imshow(uint8(X), 'InitialMagnification', 'fit');
ylabel(sprintf('Original \nMemory: %.2f KB', original_memory / 1024), 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'black', 'Rotation', 0, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right');
title('HOOI', 'FontSize', 20, 'FontWeight', 'bold', 'Color', 'black');

% Loop through different ranks
for idx = 1:length(ranks)
    R = [ranks(idx), ranks(idx), min(size(X,3), ranks(idx))];  % Define ranks for all dimensions

    % Apply HOOI to the image tensor
    [G_final, A] = HOOI(X, R, 1000, 0.0001);

    % Reconstruct the image using the core tensor and factor matrices
    X_reconstructed = tprod(G_final, A);

    % Convert back to uint8 for display if necessary
    X_reconstructed_uint8 = uint8(max(min(X_reconstructed, 255), 0));

    % Calculate memory usage for this rank
    memory_usages(idx) = numel(G_final) * 8 + sum(cellfun(@(x) numel(x), A)) * 8;

    % Calculate loss using Frobenius norm and convert to percentage
    losses(idx) = 100 * (norm(X(:) - X_reconstructed(:), 'fro') / norm(X(:), 'fro'));

    % Display the reconstructed image
    hSub(idx+1) = subplot(length(ranks)+1, 1, idx+1);
    imshow(X_reconstructed_uint8, 'InitialMagnification', 'fit');
    % Add memory and loss data to the ylabel
    ylabel(sprintf('Rank %d, Rank %d, Rank %d\nMemory: %.2f KB\nLoss: %.4f%%', R(1), R(2), R(3), memory_usages(idx)/1024, losses(idx)), 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'red', 'Rotation', 0, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right');
end

% Adjust spacing between subplots
for i = 1:length(hSub)
    pos = get(hSub(i), 'Position');
    pos(2) = pos(2) - 0.01 * i;  % Slightly move each subplot up to tighten spacing
    pos(4) = pos(4) * 1.05;  % Slightly increase the height of each subplot
    set(hSub(i), 'Position', pos);
end

% Display the memory usages and losses
disp('Memory Usages (bytes):');
disp(memory_usages);
disp('Relative Losses:');
disp(losses);

% Plotting the results
figure;
subplot(2,1,1); % Subplot for memory usage
plot(ranks, memory_usages / 1024, '-o');  % Convert bytes to kilobytes
title('Memory Usage by Rank');
xlabel('Rank');
ylabel('Memory Usage (KB)');

subplot(2,1,2); % Subplot for losses
plot(ranks, losses, '-o', 'Color', 'r');
title('Relative Loss by Rank');
xlabel('Rank');
ylabel('Relative Loss (%)');
