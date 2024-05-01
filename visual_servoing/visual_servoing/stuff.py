@staticmethod
    def get_point(img, lower_white, upper_white, epsilon, slope):
        # Pre-processing
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        edges = cv2.Canny(mask, 500, 1200)
        debug_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Work in normal polar coordinates one "distance" is 1 and one "angle" is pi/180 radians
        lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
        r, theta = lines[:,0,0], lines[:,0,1]
        c, s = np.cos(theta), np.sin(theta) + epsilon # Add to not divide by 0
        m, b = -c/s, r/s
        m_positive, b_positive = m[m > slope], b[m > slope] # Select for vertical lines
        N_positive = m_positive.shape[0]
        m_negative, b_negative = m[m < -slope], b[m < -slope] # Select for vertical lines
        N_negative = m_negative.shape[0]

        # x_intersections = []
        # y_intersections = []
        s_x, s_y, N = 0, 0, 0
        for i in range(N_positive):
            for j in range(N_negative):
                x = (b_positive[i] - b_negative[j]) / (m_negative[j] - m_positive[i])
                s_x += x
                s_y += m_positive[i]*x + b_positive[i]
                N += 1
                # x_intersections.append(x)
                # y_intersections.append(y)

        # Draw positive
        for m_,b_ in zip(m_positive, b_positive):
            x0 = -1000
            y0 = int(m_*x0 + b_)
            xf = 1000
            yf = int(m_*xf + b_)
            cv2.line(debug_rgb, (x0,y0), (xf,yf), (0, 0, 255), 2)
        # Draw negative lines
        for m_,b_ in zip(m_negative, b_negative):
            x0 = -1000
            y0 = int(m_*x0 + b_)
            xf = 1000
            yf = int(m_*xf + b_)
            cv2.line(debug_rgb, (x0,y0), (xf,yf), (0, 0, 255), 2)
        
        return debug_rgb, s_x/N, s_y/N