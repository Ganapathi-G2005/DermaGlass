import React from 'react';
import { twMerge } from 'tailwind-merge';

const GlassCard = ({ className, children, as: Component = 'div', ...rest }) => {
    return (
        <Component
            className={twMerge(
                'glass rounded-3xl shadow-[0_8px_30px_rgba(0,0,0,0.04)] border border-white/40 bg-white/70 backdrop-blur-lg',
                className
            )}
            {...rest}
        >
            {children}
        </Component>
    );
};

export default GlassCard;

