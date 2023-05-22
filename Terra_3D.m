function Terra_3D(Rt)

% Terra_3D.m - Earth texture loaded in a plot.
%
% PROTOTYPE:
%   Terra_3D(Rt)
%
% DESCRIPTION:
%   Function to load the Earth modelled as a sphere inside a figure.
%
% INPUT:
%   Rt          [1x1]       Earth mean radius       [km]
%
% OUTPUT:
%   []          [figure]    Figure open with the Earth picture loaded
%
% AUTHOR:
%   Marco Nugnes, 2020, MATLAB, Terra_3D.m
% ------------------------------------------------------------------------

%% Default Input

% Set the default value for the Earth radius in case of no inputs.
if nargin < 1
    Rt = 6371.01;                                       % [km]
end

%%  Load the Earth image from a website
% Earth_image = 'https://www.solarsystemscope.com/textures/download/8k_earth_daymap.jpg';
Earth_image = 'http://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Land_ocean_ice_2048.jpg/1024px-Land_ocean_ice_2048.jpg';

%% Figure

% Choose the color of the figure background
background_plot = 'w';

% Create the figure
figure('Color', background_plot);
hold on;
grid on;

% Set the axes scale equal
axis equal;

% Put the axes labels
xlabel('X [km]');
ylabel('Y [km]');
zlabel('Z [km]');

% Set initial view
view(120,30);

%% Create Earth surface as a wireframe

% Define the number of panels to be used to model the sphere 
npanels = 180;  

% Create a 3D meshgrid of the sphere points using the ellipsoid function
[x, y, z] = ellipsoid(0, 0, 0, Rt, Rt, Rt, npanels);

% Create the globe with the surf function
globe = surf(x, y, -z, 'FaceColor', 'none', 'EdgeColor', 'none');

%% Texturemap the globe

% Load Earth image for texture map
cdata = imread(Earth_image);

% Set the transparency of the globe: 1 = opaque, 0 = invisible
alpha = 1; 

% Set the 'FaceColor' to 'texturemap' to apply an image on the globe, and
% specify the image data using the 'CData' property with the data loaded 
% from the image. Finally, set the transparency and remove the edges.
set(globe, 'FaceColor', 'texturemap', 'CData', cdata, 'FaceAlpha', alpha, 'EdgeColor', 'none');

end

